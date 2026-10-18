"""
Scatter Search + Tabu Search para búsqueda de sub-redes (SLTH).
Implementación basada en el paper adaptado a tensores binarios.

Scatter Search es una metaheurística de población que combina:
1. Diversification Method: Genera soluciones diversas
2. Improvement Method: Mejora soluciones (Tabu Search)
3. Reference Set Update: Mantiene mejores + diversas
4. Subset Generation: Selecciona pares del RefSet
5. Solution Combination: Combina soluciones mediante XOR + reparación

Adaptación para SLTH:
- Soluciones: Máscaras binarias (lista de tensores por capa)
- Diversidad: Distancia de Hamming entre máscaras
- Combinación: XOR entre máscaras + reparación de sparsity
- Mejora local: Tabu Search con swap 1-to-1
- Sparsity: Fijo (viene del config, se preserva siempre)
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Callable, Set
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


class TabuList:
    """Lista tabú con estructura de movimientos prohibidos."""
    
    def __init__(self, tenure: int):
        """
        Args:
            tenure: Tamaño de la lista tabú (número de iteraciones que un movimiento está prohibido)
        """
        self.tenure = tenure
        self.tabu_dict = {}  # {move_hash: iteration_expires}
        self.current_iteration = 0
    
    def is_tabu(self, move: Tuple) -> bool:
        """Verifica si un movimiento está en la lista tabú."""
        move_hash = hash(move)
        if move_hash in self.tabu_dict:
            return self.tabu_dict[move_hash] > self.current_iteration
        return False
    
    def add_move(self, move: Tuple):
        """Agrega un movimiento a la lista tabú."""
        move_hash = hash(move)
        self.tabu_dict[move_hash] = self.current_iteration + self.tenure
    
    def update_iteration(self):
        """Incrementa la iteración actual y limpia movimientos expirados."""
        self.current_iteration += 1
        # Limpiar movimientos expirados
        expired = [k for k, v in self.tabu_dict.items() if v <= self.current_iteration]
        for k in expired:
            del self.tabu_dict[k]
    
    def clear(self):
        """Limpia la lista tabú."""
        self.tabu_dict.clear()


def mask_to_move(mask1: List[torch.Tensor], mask2: List[torch.Tensor]) -> Tuple:
    """
    Extrae el movimiento (swap) que diferencia dos máscaras vecinas.
    Asume que mask2 = swap_1to1(mask1).
    
    Returns:
        Tupla (layer_idx, pos_off, pos_on) representando el swap realizado
    """
    for layer_idx, (m1, m2) in enumerate(zip(mask1, mask2)):
        diff = (m1 - m2).abs()
        if diff.sum() > 0:
            # Encontrar posiciones que cambiaron
            changed = torch.nonzero(diff)
            if len(changed) >= 2:
                # Una fue apagada (1->0) y otra encendida (0->1)
                pos_off = None
                pos_on = None
                for pos in changed:
                    pos_tuple = tuple(pos.tolist())
                    if m1[pos_tuple] == 1 and m2[pos_tuple] == 0:
                        pos_off = pos_tuple
                    elif m1[pos_tuple] == 0 and m2[pos_tuple] == 1:
                        pos_on = pos_tuple
                
                if pos_off is not None and pos_on is not None:
                    return (layer_idx, pos_off, pos_on)
    
    return None


def tabu_search(model,
               mask_init: List[torch.Tensor],
               X_val, y_val,
               tenure: int = 10,
               max_iter: int = 100,
               max_no_improve: int = 30,
               seed_search: int = None,
               verbose: bool = False,
               obj_logger = None,
               iteration: int = 0,
               # Nuevo: callback para registrar cada evaluación a nivel global
               progress_callback: Callable[[int, float, float], None] = None,
               # Nuevo: contador global (lista con 1 entero para mutabilidad)
               global_eval_counter: List[int] = None) -> Tuple[List[torch.Tensor], float, Dict]:
    """
    Tabu Search para mejora local de una máscara.
    
    Algoritmo:
    1. Iniciar con máscara actual
    2. Para cada iteración:
        a. Generar vecinos mediante swap_1to1
        b. Filtrar movimientos tabú (excepto si criterio de aspiración)
        c. Seleccionar mejor vecino no-tabú
        d. Actualizar mejor global si mejora
        e. Agregar movimiento a lista tabú
    
    Args:
        model: Modelo MLP
        mask_init: Máscara inicial
        X_val, y_val: Datos de validación
        tenure: Tamaño de la lista tabú
        max_iter: Máximo de iteraciones
        max_no_improve: Parada por estancamiento
        seed_search: Semilla para reproducibilidad
        verbose: Verbosidad
        obj_logger: Logger de llamadas a función objetivo
        iteration: Iteración del algoritmo padre (para logging)
    
    Returns:
        Tupla (mejor_máscara, mejor_fitness, trace)
    """
    if seed_search is not None:
        set_seed(seed_search)
    
    # Inicializar
    current_mask = [m.clone() for m in mask_init]
    current_fitness = score(model, current_mask, X_val, y_val, metric='accuracy')
    # Actualizar contador global y callback
    if global_eval_counter is not None:
        global_eval_counter[0] += 1
        if progress_callback is not None:
            progress_callback(global_eval_counter[0], current_fitness, current_fitness)
    
    if obj_logger is not None:
        obj_logger.log_call(
            mask=current_mask,
            fitness=current_fitness,
            strategy='SS_TS',
            iteration=iteration
        )
    
    best_mask = [m.clone() for m in current_mask]
    best_fitness = current_fitness
    
    tabu_list = TabuList(tenure)
    
    trace = {
        'current_history': [current_fitness],
        'best_history': [best_fitness],
        'evaluations': [1],
        'no_improve_counter': [0]
    }
    
    num_evals = 1
    no_improve = 0
    
    for iter_ts in range(max_iter):
        # Parada por estancamiento
        if no_improve >= max_no_improve:
            if verbose:
                print(f"    TS parado por estancamiento ({max_no_improve} iters)")
            break
        
        # Generar vecinos y evaluar
        best_neighbor = None
        best_neighbor_fitness = -float('inf')
        best_neighbor_move = None
        
        # Intentar múltiples vecinos para encontrar uno no-tabú
        num_neighbors = 20
        for _ in range(num_neighbors):
            neighbor = swap_1to1(current_mask)
            # Reparar conectividad para garantizar solución factible
            neighbor = repair_mask_connectivity(neighbor, seed=seed_search)
            neighbor = ensure_input_output_path(neighbor, seed=seed_search)
            neighbor_fitness = score(model, neighbor, X_val, y_val, metric='accuracy')
            num_evals += 1
            if global_eval_counter is not None:
                global_eval_counter[0] += 1
                # Registrar cada evaluación individual (per-eval logging)
                if progress_callback is not None:
                    # best_so_far provisional incluye el mejor global conocido hasta ahora
                    provisional_best = best_fitness if best_fitness > neighbor_fitness else neighbor_fitness
                    progress_callback(global_eval_counter[0], neighbor_fitness, provisional_best)
            
            if obj_logger is not None:
                obj_logger.log_call(
                    mask=neighbor,
                    fitness=neighbor_fitness,
                    strategy='SS_TS',
                    iteration=iteration
                )
            
            # Extraer movimiento
            move = mask_to_move(current_mask, neighbor)
            
            # Criterio de aspiración: aceptar si mejora el mejor global
            aspiration = (neighbor_fitness > best_fitness)
            
            # Verificar si es tabú y no cumple aspiración
            if move is not None and tabu_list.is_tabu(move) and not aspiration:
                continue
            
            # Actualizar mejor vecino
            if neighbor_fitness > best_neighbor_fitness:
                best_neighbor = neighbor
                best_neighbor_fitness = neighbor_fitness
                best_neighbor_move = move
        
        # Si no se encontró vecino válido, terminar
        if best_neighbor is None:
            if verbose:
                print(f"    TS parado: no se encontraron vecinos no-tabú válidos")
            break
        
    # Actualizar solución actual
        current_mask = best_neighbor
        current_fitness = best_neighbor_fitness
        
        # Agregar movimiento a lista tabú
        if best_neighbor_move is not None:
            tabu_list.add_move(best_neighbor_move)
        tabu_list.update_iteration()
        
        # Actualizar mejor global
        if current_fitness > best_fitness:
            best_mask = [m.clone() for m in current_mask]
            best_fitness = current_fitness
            no_improve = 0
        else:
            no_improve += 1

        # Callback después de procesar iteración (usa best_fitness actualizado)
        if global_eval_counter is not None and progress_callback is not None:
            progress_callback(global_eval_counter[0], current_fitness, best_fitness)
        
        # Trace
        trace['current_history'].append(current_fitness)
        trace['best_history'].append(best_fitness)
        trace['evaluations'].append(num_evals)
        trace['no_improve_counter'].append(no_improve)
    
    if verbose:
        print(f"    TS completado: {len(trace['current_history'])} iters, "
              f"{num_evals} evals, best={best_fitness*100:.2f}%")
    
    trace['final_stats'] = {
        'iterations': len(trace['current_history']),
        'evaluations': num_evals,
        'best_fitness': best_fitness
    }
    
    return best_mask, best_fitness, trace


def hamming_distance_total(mask1: List[torch.Tensor], mask2: List[torch.Tensor]) -> int:
    """
    Calcula la distancia de Hamming total entre dos máscaras.
    
    Returns:
        Número de posiciones que difieren
    """
    total_diff = 0
    for m1, m2 in zip(mask1, mask2):
        total_diff += (m1 != m2).sum().item()
    return total_diff


def combine_masks(mask1: List[torch.Tensor], 
                 mask2: List[torch.Tensor],
                 sparsity: float) -> List[torch.Tensor]:
    """
    Combina dos máscaras mediante promedio y reparación de sparsity.
    
    Algoritmo simple del paper:
    1. Promediar las dos máscaras (valores en [0, 1])
    2. Seleccionar top-k conexiones para mantener sparsity objetivo
    
    Args:
        mask1, mask2: Máscaras a combinar
        sparsity: Sparsity objetivo (fijo, viene del config)
    
    Returns:
        Nueva máscara combinada con sparsity exacta
    """
    from mask import create_mask
    
    combined = []
    
    for m1, m2 in zip(mask1, mask2):
        # Promedio de las dos máscaras
        avg = (m1.float() + m2.float()) / 2.0
        
        # Seleccionar top-k para mantener sparsity
        shape = m1.shape
        num_active = int(m1.numel() * sparsity)
        
        # Flatten y obtener top-k
        avg_flat = avg.flatten()
        _, indices = torch.topk(avg_flat, num_active)
        
        # Crear nueva máscara
        new_mask = torch.zeros_like(avg_flat)
        new_mask[indices] = 1.0
        new_mask = new_mask.reshape(shape)
        
        combined.append(new_mask)
    
    # Reparar conectividad de la máscara combinada
    combined = repair_mask_connectivity(combined)
    combined = ensure_input_output_path(combined)
    
    return combined
    
    # Si no hay diferencias, retornar start
    if len(diff_positions) == 0:
        fitness = score(model, current, X_val, y_val, metric='accuracy')
        return current, fitness
    
    # Limitar número de pasos si se especifica
    if max_steps is not None and max_steps < len(diff_positions):
        np.random.shuffle(diff_positions)
        diff_positions = diff_positions[:max_steps]
    
    # Evaluar máscara inicial
    best_mask = [m.clone() for m in current]
    best_fitness = score(model, best_mask, X_val, y_val, metric='accuracy')
    
    if obj_logger is not None:
        obj_logger.log_call(
            mask=best_mask,
            fitness=best_fitness,
            strategy='SS_PR',
            iteration=iteration
        )
    
    # Recorrer el camino
    for layer_idx, pos in diff_positions:
        # Cambiar posición actual hacia la objetivo
        current[layer_idx][pos] = mask_end[layer_idx][pos]
        
        # Evaluar
        fitness = score(model, current, X_val, y_val, metric='accuracy')
        
        if obj_logger is not None:
            obj_logger.log_call(
                mask=current,
                fitness=fitness,
                strategy='SS_PR',
                iteration=iteration
            )
        
        # Actualizar mejor
        if fitness > best_fitness:
            best_mask = [m.clone() for m in current]
            best_fitness = fitness
    
    return best_mask, best_fitness


def generate_diverse_solution(model,
                              mask_shapes: List[Tuple[int, int]],
                              sparsity: float,
                              existing_population: List[List[torch.Tensor]],
                              X_val, y_val,
                              min_diversity: int,
                              max_attempts: int = 100,
                              seed: int = None) -> List[torch.Tensor]:
    """
    Genera una solución diversa respecto a la población existente.
    
    Args:
        model: Modelo MLP
        mask_shapes: Formas de las máscaras
        sparsity: Sparsity objetivo
        existing_population: Población existente
        X_val, y_val: Datos de validación
        min_diversity: Diversidad mínima requerida (distancia de Hamming)
        max_attempts: Máximo de intentos
        seed: Semilla
    
    Returns:
        Nueva máscara diversa
    """
    from mask import create_mask
    
    if seed is not None:
        set_seed(seed)
    
    for attempt in range(max_attempts):
        # Generar candidato aleatorio
        candidate = create_mask(mask_shapes, sparsity=sparsity, per_layer=True, 
                               seed=seed + attempt if seed else None)
        # Reparar conectividad para garantizar solución factible
        candidate = repair_mask_connectivity(candidate, seed=seed + attempt if seed else None)
        candidate = ensure_input_output_path(candidate, seed=seed + attempt if seed else None)
        
        # Verificar diversidad con población existente
        if len(existing_population) == 0:
            return candidate
        
        # Calcular diversidad mínima con población
        min_dist = min(hamming_distance_total(candidate, other) 
                      for other in existing_population)
        
        if min_dist >= min_diversity:
            return candidate
    
    # Si no se logró suficiente diversidad, retornar la última generada
    # (mejor que fallar)
    return candidate


def scatter_search(model,
                  mask_init: List[torch.Tensor],
                  X_val, y_val,
                  X_test, y_test,
                  refset_size: int = 10,
                  refset_quality_ratio: float = 0.5,
                  diversification_size: int = 30,
                  ts_tenure: int = 7,
                  ts_max_iter: int = 50,
                  ts_max_no_improve: int = 15,
                  max_iter: int = 50,
                  max_evals: int = 10000,
                  max_no_improve: int = 20,
                  min_diversity: int = None,
                  seed_search: int = None,
                  verbose: bool = True,
                  obj_logger = None) -> Tuple[List[torch.Tensor], Dict]:
    """
    Scatter Search + Tabu Search para búsqueda de sub-redes óptimas.
    
    Algoritmo:
    1. Diversification: Generar población inicial diversa
    2. Improvement: Aplicar Tabu Search a cada solución
    3. RefSet Update: Mantener refset_size mejores + diversas
    4. Para cada iteración:
        a. Subset Generation: Seleccionar pares del RefSet
        b. Solution Combination: Combinar máscaras (promedio + reparación sparsity)
        c. Improvement: Tabu Search a nuevas soluciones
        d. RefSet Update: Actualizar RefSet con nuevas soluciones
    
    Args:
        model: Modelo MLP
        mask_init: Máscara inicial (para obtener shapes y sparsity fija)
        X_val, y_val: Datos de validación
        X_test, y_test: Datos de test (solo para evaluación final)
        refset_size: Tamaño del conjunto de referencia
        refset_quality_ratio: Proporción de soluciones de calidad (resto = diversidad)
        diversification_size: Tamaño de la población inicial
        ts_tenure: Tenure para Tabu Search
        ts_max_iter: Máximo de iteraciones de Tabu Search
        ts_max_no_improve: Parada por estancamiento en TS
        max_iter: Máximo de iteraciones principales
        max_evals: Máximo de evaluaciones totales
        max_no_improve: Parada por estancamiento global
        min_diversity: Diversidad mínima requerida (None = auto)
        seed_search: Semilla
        verbose: Verbosidad
        obj_logger: Logger de llamadas a función objetivo
    
    Returns:
        Tupla (mejor_máscara, trace_dict)
    """
    if seed_search is not None:
        set_seed(seed_search)
    
    start_time = time.time()
    
    # Extraer configuración
    mask_shapes = [m.shape for m in mask_init]
    sparsity = mask_sparsity(mask_init)
    
    # Auto-calcular diversidad mínima si no se especifica
    if min_diversity is None:
        total_connections = sum(m.numel() for m in mask_init)
        min_diversity = int(total_connections * 0.05)  # 5% de las conexiones
    
    # Calcular refset_quality_size y refset_diversity_size
    refset_quality_size = int(refset_size * refset_quality_ratio)
    refset_diversity_size = refset_size - refset_quality_size
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Scatter Search + Tabu Search iniciado")
        print(f"{'='*70}")
        print(f"RefSet: {refset_size} ({refset_quality_size} calidad + {refset_diversity_size} diversidad)")
        print(f"Población inicial: {diversification_size}")
        print(f"TS: tenure={ts_tenure}, max_iter={ts_max_iter}")
        print(f"Max iter: {max_iter}, Max evals: {max_evals}")
        print(f"Sparsity objetivo: {sparsity*100:.2f}%")
        print(f"Diversidad mínima: {min_diversity} conexiones")
        print(f"{'='*70}\n")
    
    # ========================================
    # FASE 1: DIVERSIFICATION
    # ========================================
    if verbose:
        print(f"[FASE 1] Diversification - Generando {diversification_size} soluciones...")
    
    from mask import create_mask
    
    population = []
    population_fitness = []
    
    num_evals = 0
    
    # Incluir máscara inicial
    population.append([m.clone() for m in mask_init])
    init_fitness = score(model, mask_init, X_val, y_val, metric='accuracy')
    population_fitness.append(init_fitness)
    num_evals += 1
    
    if obj_logger is not None:
        obj_logger.log_call(
            mask=mask_init,
            fitness=init_fitness,
            strategy='SS_DIV',
            iteration=0
        )
    
    # Generar población diversa
    for i in range(1, diversification_size):
        if num_evals >= max_evals:
            break
        
        # Generar solución diversa
        new_mask = generate_diverse_solution(
            model=model,
            mask_shapes=mask_shapes,
            sparsity=sparsity,
            existing_population=population,
            X_val=X_val,
            y_val=y_val,
            min_diversity=min_diversity,
            seed=seed_search + i if seed_search else None
        )
        
        # Evaluar
        fitness = score(model, new_mask, X_val, y_val, metric='accuracy')
        num_evals += 1
        
        if obj_logger is not None:
            obj_logger.log_call(
                mask=new_mask,
                fitness=fitness,
                strategy='SS_DIV',
                iteration=0
            )
        
        population.append(new_mask)
        population_fitness.append(fitness)
    
    if verbose:
        print(f"  Población generada: {len(population)} soluciones")
        print(f"  Fitness: min={min(population_fitness)*100:.2f}%, "
              f"max={max(population_fitness)*100:.2f}%, "
              f"mean={np.mean(population_fitness)*100:.2f}%")
    
    # ========================================
    # FASE 2: IMPROVEMENT
    # ========================================
    if verbose:
        print(f"\n[FASE 2] Improvement - Aplicando Tabu Search a población inicial...")
    
    improved_population = []
    improved_fitness = []
    # Historial detallado (por evaluación) del mejor global hasta crear RefSet
    dense_evaluations = []
    dense_best_history = []
    
    for i, (mask, fitness) in enumerate(zip(population, population_fitness)):
        if num_evals >= max_evals:
            break
        
        if verbose:
            print(f"  TS {i+1}/{len(population)}: fitness inicial={fitness*100:.2f}%", end=" ")
        
        # Aplicar Tabu Search
        # Preparar contador global y callback para registrar cada evaluación en TS
        global_eval_counter = [num_evals]

        def ts_progress_callback(global_eval_number: int, current_fit: float, best_fit: float):
            # Guardar historial denso preliminar (sin RefSet aún)
            dense_evaluations.append(global_eval_number)
            dense_best_history.append(best_fit)

        improved_mask, improved_fit, ts_trace = tabu_search(
            model=model,
            mask_init=mask,
            X_val=X_val,
            y_val=y_val,
            tenure=ts_tenure,
            max_iter=ts_max_iter,
            max_no_improve=ts_max_no_improve,
            seed_search=seed_search + 1000 + i if seed_search else None,
            verbose=False,
            obj_logger=obj_logger,
            iteration=0,
            progress_callback=ts_progress_callback,
            global_eval_counter=global_eval_counter
        )
        
        # Actualizar num_evals tomando el contador global
        num_evals = global_eval_counter[0]
        
        improved_population.append(improved_mask)
        improved_fitness.append(improved_fit)
        
        if verbose:
            print(f"→ final={improved_fit*100:.2f}% (+{(improved_fit-fitness)*100:.2f}%)")
    
    # ========================================
    # FASE 3: REFSET UPDATE
    # ========================================
    if verbose:
        print(f"\n[FASE 3] RefSet Update - Seleccionando {refset_size} mejores...")
    
    # Ordenar por fitness
    sorted_indices = np.argsort(improved_fitness)[::-1]
    
    # Seleccionar mejores para calidad
    refset = []
    refset_fitness = []
    
    for i in range(min(refset_quality_size, len(sorted_indices))):
        idx = sorted_indices[i]
        refset.append(improved_population[idx])
        refset_fitness.append(improved_fitness[idx])
    
    # Seleccionar diversas para diversidad
    remaining_indices = sorted_indices[refset_quality_size:]
    
    for _ in range(min(refset_diversity_size, len(remaining_indices))):
        # Encontrar la más diversa respecto al refset actual
        best_diverse_idx = None
        best_diversity = -1
        
        for idx in remaining_indices:
            # Calcular diversidad mínima con refset
            min_dist = min(hamming_distance_total(improved_population[idx], ref_mask) 
                          for ref_mask in refset)
            
            if min_dist > best_diversity:
                best_diversity = min_dist
                best_diverse_idx = idx
        
        if best_diverse_idx is not None:
            refset.append(improved_population[best_diverse_idx])
            refset_fitness.append(improved_fitness[best_diverse_idx])
            remaining_indices = [idx for idx in remaining_indices if idx != best_diverse_idx]
    
    # Mejor global inicial
    best_mask = [m.clone() for m in refset[0]]
    best_fitness = refset_fitness[0]
    best_iteration = 0
    
    if verbose:
        print(f"  RefSet inicializado con {len(refset)} soluciones")
        print(f"  Mejor fitness: {best_fitness*100:.4f}%")
    
    # Trace
    trace = {
        'best_history': [best_fitness],
        'refset_best_history': [best_fitness],
        'refset_mean_history': [np.mean(refset_fitness)],
        'evaluations': [num_evals],
        'times': [time.time() - start_time],
        'no_improve_counter': [0],
        'refset_diversity_history': [],
        # Agregar historial denso previo a la creación del RefSet (puede estar vacío)
        'dense_evaluations': dense_evaluations,
        'dense_global_best_history': dense_best_history
    }
    
    # Calcular diversidad inicial del refset
    if len(refset) > 1:
        diversities = []
        for i in range(len(refset)):
            for j in range(i+1, len(refset)):
                diversities.append(hamming_distance_total(refset[i], refset[j]))
        trace['refset_diversity_history'].append(np.mean(diversities))
    else:
        trace['refset_diversity_history'].append(0)
    
    no_improve_counter = 0
    
    # ========================================
    # FASE 4: MAIN LOOP
    # ========================================
    if verbose:
        print(f"\n[FASE 4] Main Loop - Iteraciones principales...")
        print(f"{'─'*70}\n")
    
    for iteration in range(max_iter):
        if num_evals >= max_evals:
            if verbose:
                print(f"\n Parada por máximo de evaluaciones ({max_evals})")
            break
        
        if no_improve_counter >= max_no_improve:
            if verbose:
                print(f"\n Parada por estancamiento ({no_improve_counter} iters sin mejora)")
            break
        
        if verbose:
            print(f"Iter {iteration+1}/{max_iter} | "
                  f"Best={best_fitness*100:.4f}% | "
                  f"Evals={num_evals}/{max_evals} | "
                  f"NoImpr={no_improve_counter}")
        
        # ────────────────────────────────────────
        # 4.1: SUBSET GENERATION + COMBINATION
        # ────────────────────────────────────────
        new_solutions = []
        new_fitness = []
        
        # Generar pares del refset para combinación
        num_pairs = min(len(refset) * (len(refset) - 1) // 2, 10)  # Limitar a 10 pares
        pairs_generated = 0
        
        for i in range(len(refset)):
            for j in range(i+1, len(refset)):
                if pairs_generated >= num_pairs or num_evals >= max_evals:
                    break
                
                # Combinar máscaras (promedio + reparación de sparsity)
                combined = combine_masks(refset[i], refset[j], sparsity)
                
                # Evaluar combinación
                combined_fitness = score(model, combined, X_val, y_val, metric='accuracy')
                num_evals += 1
                
                if obj_logger is not None:
                    obj_logger.log_call(
                        mask=combined,
                        fitness=combined_fitness,
                        strategy='SS_COMBINE',
                        iteration=iteration+1
                    )
                
                new_solutions.append(combined)
                new_fitness.append(combined_fitness)
                
                pairs_generated += 1
            
            if pairs_generated >= num_pairs or num_evals >= max_evals:
                break
        
        if verbose and len(new_solutions) > 0:
            print(f"  Combination: {len(new_solutions)} soluciones generadas, "
                  f"best={max(new_fitness)*100:.2f}%")
        
        # ────────────────────────────────────────
        # 4.2: IMPROVEMENT
        # ────────────────────────────────────────
        improved_new = []
        improved_new_fitness = []
        
        for sol, fit in zip(new_solutions, new_fitness):
            if num_evals >= max_evals:
                break
            
            # Aplicar Tabu Search con registro denso
            global_eval_counter = [num_evals]

            def ts_progress_callback(global_eval_number: int, current_fit: float, best_fit: float):
                # Durante el main loop ya existe RefSet, registramos mejor global denso
                trace['dense_evaluations'].append(global_eval_number)
                trace['dense_global_best_history'].append(best_fit)

            improved, improved_fit, ts_trace = tabu_search(
                model=model,
                mask_init=sol,
                X_val=X_val,
                y_val=y_val,
                tenure=ts_tenure,
                max_iter=ts_max_iter,
                max_no_improve=ts_max_no_improve,
                seed_search=seed_search + 2000 + iteration if seed_search else None,
                verbose=False,
                obj_logger=obj_logger,
                iteration=iteration+1,
                progress_callback=ts_progress_callback,
                global_eval_counter=global_eval_counter
            )
            
            num_evals = global_eval_counter[0]
            
            improved_new.append(improved)
            improved_new_fitness.append(improved_fit)
        
        if verbose and len(improved_new) > 0:
            print(f"  Improvement: {len(improved_new)} soluciones mejoradas, "
                  f"best={max(improved_new_fitness)*100:.2f}%")
        
        # ────────────────────────────────────────
        # 4.3: REFSET UPDATE
        # ────────────────────────────────────────
        # Combinar refset actual con nuevas soluciones mejoradas
        all_candidates = refset + improved_new
        all_candidates_fitness = refset_fitness + improved_new_fitness
        
        # Ordenar por fitness
        sorted_indices = np.argsort(all_candidates_fitness)[::-1]
        
        # Seleccionar nuevas mejores para calidad
        new_refset = []
        new_refset_fitness = []
        
        for i in range(min(refset_quality_size, len(sorted_indices))):
            idx = sorted_indices[i]
            new_refset.append(all_candidates[idx])
            new_refset_fitness.append(all_candidates_fitness[idx])
        
        # Seleccionar diversas
        remaining_indices = sorted_indices[refset_quality_size:]
        
        for _ in range(min(refset_diversity_size, len(remaining_indices))):
            # Encontrar la más diversa
            best_diverse_idx = None
            best_diversity = -1
            
            for idx in remaining_indices:
                if len(new_refset) == 0:
                    best_diverse_idx = idx
                    break
                
                min_dist = min(hamming_distance_total(all_candidates[idx], ref_mask) 
                              for ref_mask in new_refset)
                
                if min_dist > best_diversity:
                    best_diversity = min_dist
                    best_diverse_idx = idx
            
            if best_diverse_idx is not None:
                new_refset.append(all_candidates[best_diverse_idx])
                new_refset_fitness.append(all_candidates_fitness[best_diverse_idx])
                remaining_indices = [idx for idx in remaining_indices if idx != best_diverse_idx]
        
        # Actualizar refset
        refset = new_refset
        refset_fitness = new_refset_fitness
        
        # Actualizar mejor global
        current_best = max(refset_fitness)
        if current_best > best_fitness:
            best_idx = refset_fitness.index(current_best)
            best_mask = [m.clone() for m in refset[best_idx]]
            best_fitness = current_best
            best_iteration = iteration + 1
            no_improve_counter = 0
            
            if verbose:
                print(f"  🎯 NUEVO MEJOR: {best_fitness*100:.4f}%")
        else:
            no_improve_counter += 1
        
        # Calcular diversidad del refset
        if len(refset) > 1:
            diversities = []
            for i in range(len(refset)):
                for j in range(i+1, len(refset)):
                    diversities.append(hamming_distance_total(refset[i], refset[j]))
            avg_diversity = np.mean(diversities)
        else:
            avg_diversity = 0
        
        # Trace
        trace['best_history'].append(best_fitness)
        trace['refset_best_history'].append(max(refset_fitness))
        trace['refset_mean_history'].append(np.mean(refset_fitness))
        trace['evaluations'].append(num_evals)
        trace['times'].append(time.time() - start_time)
        trace['no_improve_counter'].append(no_improve_counter)
        trace['refset_diversity_history'].append(avg_diversity)
        
        if verbose:
            print(f"  RefSet: best={max(refset_fitness)*100:.2f}%, "
                  f"mean={np.mean(refset_fitness)*100:.2f}%, "
                  f"diversity={avg_diversity:.0f}")
            print()
    
    # ========================================
    # EVALUACIÓN FINAL EN TEST
    # ========================================
    acc_test = score(model, best_mask, X_test, y_test, metric='accuracy')
    
    total_time = time.time() - start_time
    
    # Calcular hipervolumen (para comparación con multiobjetivo)
    # f1 = 1 - accuracy (minimizar), f2 = sparsity (fijo)
    final_sparsity = mask_sparsity(best_mask)
    f1 = 1.0 - best_fitness
    f2 = final_sparsity
    hypervolume = calculate_hypervolume_single_point(f1, f2, ref_point=(1.0, 1.0))
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Scatter Search + Tabu Search finalizado")
        print(f"{'='*70}")
        print(f"Mejor accuracy validación: {best_fitness*100:.4f}%")
        print(f"Accuracy test: {acc_test*100:.4f}%")
        print(f"Hipervolumen: {hypervolume:.6f}")
        print(f"  (f1={f1:.4f}, f2={f2:.4f})")
        print(f"Encontrado en iteración: {best_iteration}")
        print(f"Iteraciones totales: {iteration + 1}")
        print(f"Evaluaciones totales: {num_evals}")
        print(f"Tiempo total: {total_time:.2f}s")
        print(f"Sparsity: {mask_sparsity(best_mask)*100:.2f}%")
        print(f"Conexiones activas: {count_active_connections(best_mask)}")
        print(f"{'='*70}\n")
    
    # Stats finales
    trace['final_stats'] = {
        'best_accuracy': best_fitness,
        'acc_test': acc_test,
        'hypervolume': hypervolume,
        'f1': f1,
        'f2': f2,
        'best_iteration': best_iteration,
        'total_evaluations': num_evals,
        'total_time': total_time,
        'iterations': iteration + 1,
        'sparsity': mask_sparsity(best_mask),
        'active_connections': count_active_connections(best_mask)
    }
    
    return best_mask, trace


def run_ss_experiment(model, mask_init, X_val, y_val, X_test, y_test,
                     hp_ss: dict, seed_search: int, verbose: bool = False,
                     obj_logger = None) -> dict:
    """
    Ejecuta un experimento completo de Scatter Search con evaluación en test.
    
    Args:
        model: Modelo MLP
        mask_init: Máscara inicial
        X_val, y_val: Validación
        X_test, y_test: Test
        hp_ss: Hiperparámetros de Scatter Search
        seed_search: Semilla de búsqueda
        verbose: Verbosidad
        obj_logger: Logger para llamadas a función objetivo
    
    Returns:
        Dict con resultados completos
    """
    # Ejecutar Scatter Search
    best_mask, trace = scatter_search(
        model=model,
        mask_init=mask_init,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        refset_size=hp_ss.get('refset_size', 10),
        refset_quality_ratio=hp_ss.get('refset_quality_ratio', 0.5),
        diversification_size=hp_ss.get('diversification_size', 30),
        ts_tenure=hp_ss.get('ts_tenure', 7),
        ts_max_iter=hp_ss.get('ts_max_iter', 50),
        ts_max_no_improve=hp_ss.get('ts_max_no_improve', 15),
        max_iter=hp_ss.get('max_iter', 50),
        max_evals=hp_ss.get('max_evals', 10000),
        max_no_improve=hp_ss.get('max_no_improve', 20),
        min_diversity=hp_ss.get('min_diversity', None),
        seed_search=seed_search,
        verbose=verbose,
        obj_logger=obj_logger
    )
    
    # Compilar resultados
    results = {
        'best_mask': best_mask,
        'acc_explore': trace['final_stats']['best_accuracy'],
        'acc_test': trace['final_stats']['acc_test'],
        'hypervolume': trace['final_stats']['hypervolume'],
        'f1': trace['final_stats']['f1'],
        'f2': trace['final_stats']['f2'],
        'trace': trace,
        'seed_search': seed_search,
        'hyperparameters': hp_ss
    }
    
    return results


if __name__ == "__main__":
    print("=== Test: Scatter Search + Tabu Search ===\n")
    
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
    
    # Hiperparámetros SS
    hp_ss = {
        'refset_size': 6,
        'refset_quality_ratio': 0.5,
        'diversification_size': 15,
        'ts_tenure': 5,
        'ts_max_iter': 30,
        'ts_max_no_improve': 10,
        'max_iter': 20,
        'max_evals': 2000,
        'max_no_improve': 10,
        'min_diversity': None
    }
    
    # Ejecutar Scatter Search
    print("\nEjecutando Scatter Search + Tabu Search...")
    print("-" * 70)
    
    results = run_ss_experiment(
        model=model,
        mask_init=mask_init,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        hp_ss=hp_ss,
        seed_search=200,
        verbose=True
    )
    
    print("\n=== Resultados Finales ===")
    print(f"Accuracy exploración: {results['acc_explore']*100:.4f}%")
    print(f"Accuracy test: {results['acc_test']*100:.4f}%")
    print(f"Iteraciones: {results['trace']['final_stats']['iterations']}")
    print(f"Evaluaciones: {results['trace']['final_stats']['total_evaluations']}")
    print(f"Tiempo: {results['trace']['final_stats']['total_time']:.2f}s")
