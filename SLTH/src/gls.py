"""
Guided Local Search (GLS) para búsqueda de sub-redes (SLTH).
Metaheurística que combina búsqueda local con penalizaciones adaptativas.

GLS penaliza características (conexiones) que aparecen frecuentemente en 
óptimos locales, guiando la búsqueda hacia regiones no exploradas del espacio.

Para SLTH:
- Características: Conexiones individuales en la máscara
- Costo base: Balance entre accuracy y sparsity
- Penalizaciones: Incrementan cuando una conexión está activa en óptimo local
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Callable
import time
import copy
from seeds import set_seed
from mask import (
    swap_1to1,
    mask_sparsity,
    count_active_connections,
    hamming_distance,
    repair_mask_connectivity,
    ensure_input_output_path,
)
from eval import score


def initialize_penalties(mask_shapes: List[Tuple[int, int]]) -> List[torch.Tensor]:
    """
    Inicializa penalizaciones a cero para cada conexión.
    
    Args:
        mask_shapes: Formas de las máscaras por capa
    
    Returns:
        Lista de tensores de penalizaciones (inicializadas a 0)
    """
    penalties = []
    for shape in mask_shapes:
        penalties.append(torch.zeros(shape))
    return penalties


def evaluate_base_cost(model, mask: List[torch.Tensor], 
                       X_val, y_val, 
                       alpha: float,
                       target_sparsity: float) -> float:
    """
    Evalúa el costo base de una máscara (sin penalizaciones).
    
     FIX: Si swap_1to1 mantiene sparsity exacta, el término de desviación
    es siempre 0. Simplificamos a solo accuracy.
    
    Costo = 1 - accuracy (a minimizar)
    
    Nota: El parámetro alpha se mantiene por compatibilidad pero no se usa
    si la sparsity se preserva exactamente (como debe ser con swap_1to1).
    
    Args:
        model: Modelo MLP
        mask: Máscara actual
        X_val, y_val: Datos de validación
        alpha: Factor de balance [0, 1] (no usado si sparsity fija)
        target_sparsity: Sparsity objetivo (no usado si sparsity fija)
    
    Returns:
        Costo base (a minimizar)
    """
    #  FIX: Solo accuracy (swap_1to1 mantiene sparsity)
    acc = score(model, mask, X_val, y_val, metric='accuracy')
    return 1.0 - acc


def evaluate_augmented_cost(model, mask: List[torch.Tensor],
                           penalties: List[torch.Tensor],
                           X_val, y_val,
                           alpha: float,
                           lambda_gls: float,
                           target_sparsity: float) -> Tuple[float, float]:
    """
    Evalúa el costo aumentado (base + penalizaciones).
    
    h(m) = f(m) + λ * Σ(p_i * I_i(m))
    
    donde:
    - f(m): costo base
    - p_i: penalización de la conexión i
    - I_i(m): indicador (1 si conexión i está activa, 0 si no)
    - λ: factor de penalización
    
    Args:
        model: Modelo MLP
        mask: Máscara actual
        penalties: Penalizaciones por conexión
        X_val, y_val: Datos de validación
        alpha: Balance entre loss y sparsity
        lambda_gls: Factor de penalización
        target_sparsity: Sparsity objetivo
    
    Returns:
        Tupla (costo_aumentado, costo_base)
    """
    # Costo base
    base_cost = evaluate_base_cost(model, mask, X_val, y_val, alpha, target_sparsity)
    
    # Penalización: suma de penalizaciones de conexiones activas
    penalty_term = 0.0
    for mask_layer, penalty_layer in zip(mask, penalties):
        # Solo penalizar conexiones activas (mask == 1)
        active_penalties = penalty_layer * mask_layer
        penalty_term += active_penalties.sum().item()
    
    # Costo aumentado
    augmented_cost = base_cost + lambda_gls * penalty_term
    
    return augmented_cost, base_cost


def local_search_first_improvement(model, 
                                   mask_init: List[torch.Tensor],
                                   penalties: List[torch.Tensor],
                                   X_val, y_val,
                                   alpha: float,
                                   lambda_gls: float,
                                   target_sparsity: float,
                                   max_neighbors: int = 100,
                                   max_local_iters: int = 10,
                                   iteration: int = 0,
                                   obj_logger = None) -> Tuple[List[torch.Tensor], float, int]:
    """
     OPTIMIZADO: Búsqueda local first-improvement con límite de ciclos.
    
    Intenta MÚLTIPLES vecinos hasta encontrar el PRIMERO que mejore,
    luego reinicia desde ese mejor. Continúa hasta agotar presupuesto
    o no hallar mejoras.
    
    Optimización: Ahora tiene DOS límites:
    1. max_neighbors: Total de evaluaciones permitidas
    2. max_local_iters: Máximo de ciclos de mejora (evita LS muy largas)
    
    Args:
        model: Modelo MLP
        mask_init: Máscara inicial
        penalties: Penalizaciones actuales
        X_val, y_val: Datos de validación
        alpha, lambda_gls, target_sparsity: Parámetros de costo
        max_neighbors: Máximo de vecinos a explorar
        max_local_iters: Máximo de ciclos de mejora (nuevo)
        iteration: Iteración GLS actual (para logging)
        obj_logger: Logger para llamadas a función objetivo (opcional)
    
    Returns:
        Tupla (mejor_máscara, mejor_costo_aumentado, num_evaluaciones)
    """
    current_mask = [m.clone() for m in mask_init]
    current_cost, _ = evaluate_augmented_cost(
        model, current_mask, penalties, X_val, y_val, 
        alpha, lambda_gls, target_sparsity
    )
    
    # Log evaluación inicial (solo si hay logger para evitar overhead)
    if obj_logger is not None:
        current_acc = score(model, current_mask, X_val, y_val, metric='accuracy')
        obj_logger.log_call(
            mask=current_mask,
            fitness=current_acc,
            strategy='GLS_LS',
            iteration=iteration
        )
    
    evaluations = 1  # Contamos la evaluación inicial
    local_iter = 0  #  Contador de ciclos de mejora
    
    #  OPTIMIZACIÓN: Límite de ciclos + límite de evaluaciones
    while evaluations < max_neighbors and local_iter < max_local_iters:
        improved = False
        attempts = 0  #  Intentos en este ciclo
        max_attempts = min(50, max_neighbors - evaluations)  #  Límite por ciclo
        
        for _ in range(max_attempts):
            # Generar vecino
            neighbor = swap_1to1(current_mask)
            neighbor_cost, _ = evaluate_augmented_cost(
                model, neighbor, penalties, X_val, y_val,
                alpha, lambda_gls, target_sparsity
            )
            evaluations += 1
            attempts += 1
            
            #  First improvement: acepta el PRIMER vecino mejor
            if neighbor_cost < current_cost - 1e-12:  # Tolerancia numérica
                current_mask = neighbor
                current_cost = neighbor_cost
                improved = True
                
                #  Log solo mejoras (reduce overhead)
                if obj_logger is not None:
                    neighbor_acc = score(model, neighbor, X_val, y_val, metric='accuracy')
                    obj_logger.log_call(
                        mask=neighbor,
                        fitness=neighbor_acc,
                        strategy='GLS_LS',
                        iteration=iteration
                    )
                break  # Encontró mejora, reinicia búsqueda desde aquí
            
            # Si agotamos presupuesto
            if evaluations >= max_neighbors:
                break
        
        # Si no encontró mejora en este ciclo → óptimo local alcanzado
        if not improved:
            break
        
        local_iter += 1  #  Incrementar contador de ciclos
    
    return current_mask, current_cost, evaluations


def update_penalties(mask: List[torch.Tensor],
                    penalties: List[torch.Tensor],
                    costs: List[torch.Tensor] = None) -> List[torch.Tensor]:
    """
     DEPRECATED: Penaliza TODAS las conexiones activas (incorrecto para GLS).
    Usa update_penalties_max_utility() en su lugar.
    
    Se mantiene por compatibilidad.
    """
    new_penalties = [p.clone() for p in penalties]
    
    for layer_idx, (mask_layer, penalty_layer) in enumerate(zip(mask, new_penalties)):
        # Incrementar penalización solo para conexiones activas
        if costs is None:
            # Costos constantes = 1
            penalty_layer += mask_layer.float()
        else:
            # Costos variables
            penalty_layer += mask_layer.float() * costs[layer_idx]
    
    return new_penalties


def update_penalties_max_utility(mask: List[torch.Tensor],
                                 penalties: List[torch.Tensor],
                                 costs: List[torch.Tensor] = None,
                                 by_layer: bool = True) -> Tuple[List[torch.Tensor], int]:
    """
     FIX: Penaliza SOLO los rasgos activos de MÁXIMA UTILIDAD.
    
    GLS clásico: u_i = c_i / (1 + p_i)
    Selecciona S = {i : u_i = max u_j} y penaliza solo esos.
    
    Args:
        mask: Máscara del óptimo local
        penalties: Penalizaciones actuales
        costs: Costos por conexión (None = constantes c_i = 1)
        by_layer: Si True, selecciona máxima utilidad por capa
    
    Returns:
        Tupla (penalizaciones_actualizadas, total_conexiones_penalizadas)
    """
    new_penalties = [p.clone() for p in penalties]
    
    if costs is None:
        # Costos constantes c_i = 1
        costs = [torch.ones_like(p, dtype=torch.float32) for p in penalties]
    
    total_selected = 0
    
    for layer_idx, (m_layer, p_layer, c_layer) in enumerate(zip(mask, new_penalties, costs)):
        # Solo considerar conexiones activas
        active = (m_layer == 1)
        
        if active.sum() == 0:
            continue  # Capa sin conexiones activas
        
        # Calcular utilidades: u_i = c_i / (1 + p_i)
        utilities = torch.zeros_like(p_layer, dtype=torch.float32)
        utilities[active] = c_layer[active] / (1.0 + p_layer[active])
        
        # Encontrar máxima utilidad
        u_max = utilities.max()
        
        # Seleccionar SOLO las de máxima utilidad (incluir empates)
        selected = (utilities >= u_max - 1e-12) & active
        
        # Penalizar SOLO las seleccionadas
        p_layer[selected] += c_layer[selected]
        
        total_selected += int(selected.sum().item())
    
    return new_penalties, total_selected


def calculate_hypervolume_single_point(f1: float, f2: float, 
                                       ref_point: Tuple[float, float] = (1.0, 1.0)) -> float:
    """
    Calcula el hipervolumen para un único punto en espacio bi-objetivo.
    Usado para métodos mono-objetivo con sparsity fija.
    
    Args:
        f1: Objetivo 1 (1 - accuracy, a minimizar)
        f2: Objetivo 2 (sparsity, fijo)
        ref_point: Punto de referencia (f1_ref, f2_ref)
    
    Returns:
        Hipervolumen (área dominada por el punto)
    """
    f1_ref, f2_ref = ref_point
    
    # Verificar que el punto domina al punto de referencia
    if f1 >= f1_ref or f2 >= f2_ref:
        return 0.0
    
    # HV = (f1_ref - f1) * (f2_ref - f2)
    hypervolume = (f1_ref - f1) * (f2_ref - f2)
    
    return hypervolume


def guided_local_search(model,
                       mask_init: List[torch.Tensor],
                       X_val, y_val,
                       alpha: float = 0.5,
                       lambda_gls: float = 0.1,
                       max_iter: int = 1000,
                       max_evals: int = 10000,
                       max_no_improve: int = None,
                       ls_max_neighbors: int = 100,
                       ls_max_iters: int = 10,
                       seed_search: int = None,
                       verbose: bool = True,
                       obj_logger = None) -> Tuple[List[torch.Tensor], Dict]:
    """
    Guided Local Search para búsqueda de sub-red óptima.
    
    Algoritmo:
    1. Inicializar penalizaciones p_i = 0
    2. Repetir hasta criterio de parada:
        a. Búsqueda local → encontrar óptimo local
        b. Si es mejor que el mejor global → actualizar
        c. Penalizar conexiones activas en óptimo local
        d. Reiniciar búsqueda local con costos aumentados
    
    Args:
        model: Modelo MLP congelado
        mask_init: Máscara inicial
        X_val, y_val: Datos de validación
        alpha: Balance entre loss y sparsity [0, 1]
            - 1.0 = solo accuracy
            - 0.0 = solo sparsity
        lambda_gls: Factor de penalización [0.01, 1.0]
        max_iter: Máximo de iteraciones GLS
        max_evals: Máximo de evaluaciones totales
        max_no_improve: Máximo iteraciones sin mejora (None = sin límite)
        ls_max_neighbors: Máximo de vecinos por búsqueda local
        ls_max_iters: Máximo de ciclos de mejora en búsqueda local
        seed_search: Semilla para búsqueda
        verbose: Imprimir progreso
        obj_logger: Logger para llamadas a función objetivo (opcional)
    
    Returns:
        Tupla (mejor_máscara, trace_dict)
    """
    if seed_search is not None:
        set_seed(seed_search)
    
    # Inicialización
    current_mask = [m.clone() for m in mask_init]
    target_sparsity = mask_sparsity(mask_init)
    
    # Inicializar penalizaciones
    mask_shapes = [m.shape for m in mask_init]
    penalties = initialize_penalties(mask_shapes)
    
    # Evaluar máscara inicial
    current_cost, current_base_cost = evaluate_augmented_cost(
        model, current_mask, penalties, X_val, y_val,
        alpha, lambda_gls, target_sparsity
    )
    current_acc = score(model, current_mask, X_val, y_val, metric='accuracy')
    
    # Log evaluación inicial
    if obj_logger is not None:
        obj_logger.log_call(
            mask=current_mask,
            fitness=current_acc,
            strategy='GLS',
            iteration=0
        )
    
    # Mejor global
    best_mask = [m.clone() for m in current_mask]
    best_base_cost = current_base_cost
    best_acc = current_acc
    
    # Tracking
    trace = {
        'augmented_cost_history': [current_cost],
        'base_cost_history': [current_base_cost],
        'accuracy_history': [current_acc],
        'best_accuracy_history': [best_acc],
        'penalty_sum_history': [sum(p.sum().item() for p in penalties)],
        'local_optima_found': [],
        'evaluations': [1],
        'times': [0.0],
        'no_improve_counter': [0]
    }
    
    num_evals = 1
    no_improve_counter = 0
    start_time = time.time()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Guided Local Search iniciado")
        print(f"{'='*70}")
        print(f"Alpha: {alpha}, Lambda: {lambda_gls}")
        print(f"Max iter: {max_iter}, Max evals: {max_evals}")
        print(f"LS max neighbors: {ls_max_neighbors}, LS max iters: {ls_max_iters}")
        print(f"Target sparsity: {target_sparsity*100:.2f}%")
        print(f"Accuracy inicial: {current_acc*100:.4f}%")
        print(f"{'='*70}\n")
    
    # Bucle principal GLS
    for iteration in range(max_iter):
        if num_evals >= max_evals:
            if verbose:
                print(f"\nParada por máximo de evaluaciones ({max_evals})")
            break
        
        if max_no_improve is not None and no_improve_counter >= max_no_improve:
            if verbose:
                print(f"\nParada por estancamiento ({no_improve_counter} iters sin mejora)")
            break
        
        # 1. Búsqueda local desde posición actual
        local_opt_mask, local_opt_aug_cost, ls_evals = local_search_first_improvement(
            model, current_mask, penalties, X_val, y_val,
            alpha, lambda_gls, target_sparsity, ls_max_neighbors,
            max_local_iters=ls_max_iters,
            iteration=iteration,
            obj_logger=obj_logger
        )
        num_evals += ls_evals
        
        # Evaluar óptimo local (costo base, sin penalizaciones)
        _, local_opt_base_cost = evaluate_augmented_cost(
            model, local_opt_mask, penalties, X_val, y_val,
            alpha, lambda_gls, target_sparsity
        )
        local_opt_acc = score(model, local_opt_mask, X_val, y_val, metric='accuracy')
        
        # 2. Actualizar mejor global (basado en costo base, no aumentado)
        if local_opt_base_cost < best_base_cost:
            best_mask = [m.clone() for m in local_opt_mask]
            best_base_cost = local_opt_base_cost
            best_acc = local_opt_acc
            no_improve_counter = 0
            
            if verbose:
                print(f" Iter {iteration:3d} | NUEVO MEJOR: Acc={best_acc*100:.4f}% | "
                      f"Cost={best_base_cost:.6f} | Evals={num_evals}")
        else:
            no_improve_counter += 1
        
        # 3.  FIX: Penalizar SOLO conexiones de máxima utilidad (GLS clásico)
        penalties, num_penalized = update_penalties_max_utility(
            local_opt_mask, penalties, costs=None, by_layer=True
        )
        
        # 4. Reiniciar desde óptimo local con nuevas penalizaciones
        current_mask = local_opt_mask
        current_cost = local_opt_aug_cost
        current_base_cost = local_opt_base_cost
        current_acc = local_opt_acc
        
        # Tracking
        trace['augmented_cost_history'].append(current_cost)
        trace['base_cost_history'].append(current_base_cost)
        trace['accuracy_history'].append(current_acc)
        trace['best_accuracy_history'].append(best_acc)
        trace['penalty_sum_history'].append(sum(p.sum().item() for p in penalties))
        trace['num_penalized_history'] = trace.get('num_penalized_history', [])
        trace['num_penalized_history'].append(num_penalized)
        trace['local_optima_found'].append(iteration)
        trace['evaluations'].append(num_evals)
        trace['times'].append(time.time() - start_time)
        trace['no_improve_counter'].append(no_improve_counter)
        
        # Verbose periódico
        if verbose and (iteration + 1) % 10 == 0:
            elapsed = time.time() - start_time
            evals_per_sec = num_evals / elapsed if elapsed > 0 else 0
            penalty_sum = sum(p.sum().item() for p in penalties)
            
            print(f"Iter {iteration+1:3d} | "
                  f"LocalOpt Acc={current_acc*100:.4f}% | "
                  f"Best={best_acc*100:.4f}% | "
                  f"Penalties={penalty_sum:.0f} (penalized={num_penalized}) | "
                  f"Evals={num_evals}/{max_evals} ({evals_per_sec:.1f}/s) | "
                  f"NoImpr={no_improve_counter}")
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Guided Local Search finalizado")
        print(f"{'='*70}")
        print(f"Mejor accuracy: {best_acc*100:.4f}%")
        print(f"Iteraciones GLS: {iteration + 1}")
        print(f"Evaluaciones totales: {num_evals}")
        print(f"Tiempo total: {total_time:.2f}s")
        print(f"Sparsity: {mask_sparsity(best_mask)*100:.2f}%")
        print(f"Conexiones activas: {count_active_connections(best_mask)}")
        print(f"Penalizaciones finales: {sum(p.sum().item() for p in penalties):.0f}")
        print(f"{'='*70}\n")
    
    # Stats finales - calcular hipervolumen para comparar con multi-objetivo
    final_sparsity = mask_sparsity(best_mask)
    f1 = 1.0 - best_acc  # Minimizar (1-accuracy)
    f2 = final_sparsity  # Minimizar sparsity
    final_hv = calculate_hypervolume_single_point(f1, f2, ref_point=(1.0, 1.0))
    
    trace['final_stats'] = {
        'best_accuracy': best_acc,
        'best_base_cost': best_base_cost,
        'total_evaluations': num_evals,
        'total_time': total_time,
        'iterations': iteration + 1,
        'sparsity': final_sparsity,
        'active_connections': count_active_connections(best_mask),
        'final_penalty_sum': sum(p.sum().item() for p in penalties),
        'hypervolume': final_hv,
        'f1': f1,
        'f2': f2
    }
    
    return best_mask, trace


def run_gls_experiment(model, mask_init, X_val, y_val, X_test, y_test,
                      hp_gls: dict, seed_search: int, verbose: bool = False,
                      obj_logger = None) -> dict:
    """
    Ejecuta un experimento completo de GLS con evaluación en test.
    
    Args:
        model: Modelo MLP
        mask_init: Máscara inicial
        X_val, y_val: Validación
        X_test, y_test: Test
        hp_gls: Hiperparámetros de GLS
        seed_search: Semilla de búsqueda
        verbose: Verbosidad
        obj_logger: Logger para llamadas a función objetivo (opcional)
    
    Returns:
        Dict con resultados completos
    """
    # Ejecutar GLS
    best_mask, trace = guided_local_search(
        model=model,
        mask_init=mask_init,
        X_val=X_val,
        y_val=y_val,
        alpha=hp_gls.get('alpha', 0.5),
        lambda_gls=hp_gls.get('lambda', 0.1),
        max_iter=hp_gls.get('max_iter', 1000),
        max_evals=hp_gls.get('max_evals', 10000),
        max_no_improve=hp_gls.get('max_no_improve', None),
        ls_max_neighbors=hp_gls.get('ls_max_neighbors', 100),
        ls_max_iters=hp_gls.get('ls_max_iters', 10),
        seed_search=seed_search,
        verbose=verbose,
        obj_logger=obj_logger
    )
    
    # Evaluar en test (una sola vez)
    acc_test = score(model, best_mask, X_test, y_test, metric='accuracy')
    
    # Compilar resultados
    results = {
        'best_mask': best_mask,
        'acc_explore': trace['final_stats']['best_accuracy'],
        'acc_test': acc_test,
        'hypervolume': trace['final_stats']['hypervolume'],
        'f1': trace['final_stats']['f1'],
        'f2': trace['final_stats']['f2'],
        'trace': trace,
        'seed_search': seed_search,
        'hyperparameters': hp_gls
    }
    
    return results


if __name__ == "__main__":
    print("=== Test: Guided Local Search ===\n")
    
    from model import build_mlp
    from mask import create_mask
    from data_loader import load_wine_data
    
    # Cargar datos
    print("Cargando datos Wine...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_wine_data(seed_split=300)
    
    # Crear modelo
    print("\nCreando modelo...")
    arch = [13, 128, 128, 3]
    model = build_mlp(arch, seed_model=100)
    
    # Crear máscara inicial
    print("\nCreando máscara inicial (20% sparsity)...")
    shapes = model.get_weight_shapes()
    mask_init = create_mask(shapes, sparsity=0.2, per_layer=True, seed=42)
    
    # Hiperparámetros GLS
    hp_gls = {
        'alpha': 0.9,  # Priorizar accuracy sobre sparsity
        'lambda': 0.1,  # Factor de penalización moderado
        'max_iter': 100,  # Reducido para test
        'max_evals': 2000,
        'max_no_improve': 20,
        'ls_max_neighbors': 50,
        'ls_max_iters': 5  #  Límite de ciclos de mejora
    }
    
    # Ejecutar GLS
    print("\nEjecutando Guided Local Search...")
    print("-" * 70)
    
    results = run_gls_experiment(
        model=model,
        mask_init=mask_init,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        hp_gls=hp_gls,
        seed_search=200,
        verbose=True
    )
    
    print("\n=== Resultados Finales ===")
    print(f"Accuracy exploración: {results['acc_explore']*100:.4f}%")
    print(f"Accuracy test: {results['acc_test']*100:.4f}%")
    print(f"Iteraciones GLS: {results['trace']['final_stats']['iterations']}")
    print(f"Evaluaciones: {results['trace']['final_stats']['total_evaluations']}")
    print(f"Tiempo: {results['trace']['final_stats']['total_time']:.2f}s")
    print(f"Penalizaciones finales: {results['trace']['final_stats']['final_penalty_sum']:.0f}")
