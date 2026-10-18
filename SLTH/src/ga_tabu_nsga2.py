"""
Genetic Algorithm + Tabu Search + NSGA-II para búsqueda de sub-redes (SLTH).
Metaheurística híbrida multiobjetivo usando dominancia de Pareto.

Características:
- Objetivos (ambos de minimización):
  * f1 = 1 - accuracy (minimizar error)
  * f2 = sparsity (minimizar conexiones podadas)
- Sparsity variable (no fija como en otros métodos)
- Selección por dominancia de Pareto (NSGA-II)
- Crowding distance para preservar diversidad
- Crossover: Recombinación de máscaras (media + binarización)
- Mutación: Flip de bits aleatorio con probabilidad μ
- Mejora local: Tabu Search cada N generaciones
- Elitismo: Preservar frente de Pareto
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Callable, Set
import time
import copy
import os
import csv
import matplotlib.pyplot as plt
from seeds import set_seed
from mask import (
    create_mask,
    mask_sparsity,
    count_active_connections,
    hamming_distance,
    swap_1to1,
    repair_mask_connectivity,
    ensure_input_output_path,
)
from eval import score


class TabuList:
    """Lista tabú para el componente de búsqueda local."""
    
    def __init__(self, tenure: int):
        """
        Args:
            tenure: Tamaño de la lista tabú (iteraciones que un movimiento está prohibido)
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
        expired = [k for k, v in self.tabu_dict.items() if v <= self.current_iteration]
        for k in expired:
            del self.tabu_dict[k]
    
    def clear(self):
        """Limpia la lista tabú."""
        self.tabu_dict.clear()


class Individual:
    """Representa un individuo en la población del GA con NSGA-II."""
    
    def __init__(self, mask: List[torch.Tensor]):
        """
        Args:
            mask: Lista de máscaras binarias por capa
        """
        self.mask = [m.clone() for m in mask]
        self.accuracy = None
        self.sparsity = None
        self.f1 = None  # Objetivo 1: 1 - accuracy (minimizar)
        self.f2 = None  # Objetivo 2: sparsity (minimizar)
        
        # Atributos NSGA-II
        self.domination_count = 0  # Número de individuos que lo dominan
        self.dominated_solutions = []  # Individuos que domina
        self.rank = None  # Frente de Pareto (1 = mejor)
        self.crowding_distance = 0.0  # Distancia de crowding
    
    def evaluate(self, model, X_val, y_val):
        """
        Evalúa los dos objetivos.
        
        Args:
            model: Modelo MLP
            X_val, y_val: Datos de validación
        """
        self.accuracy = score(model, self.mask, X_val, y_val, metric='accuracy')
        self.sparsity = mask_sparsity(self.mask)
        
        # Objetivos de minimización
        self.f1 = 1.0 - self.accuracy  # Minimizar error
        self.f2 = self.sparsity         # Minimizar sparsity
    
    def dominates(self, other) -> bool:
        """
        Verifica si este individuo domina a otro en el sentido de Pareto.
        Domina si es mejor o igual en todos los objetivos y estrictamente mejor en al menos uno.
        
        Objetivos (ambos minimización):
        - f1 = 1 - accuracy (minimizar)
        - f2 = sparsity (minimizar)
        """
        if self.f1 is None or other.f1 is None:
            return False
        
        # Mejor o igual en ambos objetivos (menor es mejor)
        f1_better_or_equal = self.f1 <= other.f1
        f2_better_or_equal = self.f2 <= other.f2
        
        # Estrictamente mejor en al menos uno
        f1_strictly_better = self.f1 < other.f1
        f2_strictly_better = self.f2 < other.f2
        
        return (f1_better_or_equal and f2_better_or_equal and 
                (f1_strictly_better or f2_strictly_better))


def initialize_population(mask_shapes: List[Tuple[int, int]], 
                         pop_size: int,
                         min_sparsity: float = 0.1,
                         max_sparsity: float = 0.9,
                         seed: int = None) -> List[Individual]:
    """
    Inicializa población con individuos de sparsities variadas.
    
    Args:
        mask_shapes: Formas de las máscaras por capa
        pop_size: Tamaño de la población
        min_sparsity: Sparsity mínima
        max_sparsity: Sparsity máxima
        seed: Semilla aleatoria
    
    Returns:
        Lista de individuos
    """
    if seed is not None:
        set_seed(seed)
    
    population = []
    
    # Generar individuos con diferentes sparsities
    sparsities = np.linspace(min_sparsity, max_sparsity, pop_size)
    
    for i, target_sparsity in enumerate(sparsities):
        mask = create_mask(
            mask_shapes,
            sparsity=target_sparsity,
            per_layer=True,
            seed=seed + i if seed is not None else None,
        )
        # Reparar conectividad básica para evitar neuronas completamente aisladas
        mask = repair_mask_connectivity(mask, seed=seed + i if seed is not None else None)
        # Garantizar al menos un camino completo input→output
        mask = ensure_input_output_path(mask, seed=seed + i if seed is not None else None)
        individual = Individual(mask)
        population.append(individual)
    
    return population


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """
    Algoritmo de ordenamiento rápido no dominado (NSGA-II).
    Clasifica la población en frentes de Pareto.
    
    Returns:
        Lista de frentes, donde fronts[0] es el frente de Pareto
    """
    # Resetear contadores
    for ind in population:
        ind.domination_count = 0
        ind.dominated_solutions = []
    
    fronts = [[]]  # fronts[0] = primer frente de Pareto
    
    # Calcular relaciones de dominancia
    for i, ind_i in enumerate(population):
        for j, ind_j in enumerate(population):
            if i == j:
                continue
            
            if ind_i.dominates(ind_j):
                ind_i.dominated_solutions.append(ind_j)
            elif ind_j.dominates(ind_i):
                ind_i.domination_count += 1
        
        # Si no es dominado por nadie, pertenece al primer frente
        if ind_i.domination_count == 0:
            ind_i.rank = 1
            fronts[0].append(ind_i)
    
    # Construir frentes subsecuentes
    front_idx = 0
    while len(fronts[front_idx]) > 0:
        next_front = []
        
        for ind in fronts[front_idx]:
            for dominated_ind in ind.dominated_solutions:
                dominated_ind.domination_count -= 1
                if dominated_ind.domination_count == 0:
                    dominated_ind.rank = front_idx + 2
                    next_front.append(dominated_ind)
        
        front_idx += 1
        fronts.append(next_front)
    
    # Eliminar último frente vacío
    if len(fronts[-1]) == 0:
        fronts.pop()
    
    return fronts


def calculate_crowding_distance(front: List[Individual]):
    """
    Calcula la distancia de crowding para cada individuo en un frente.
    Mide qué tan "aislado" está un individuo de sus vecinos.
    Individuos en los extremos reciben distancia infinita.
    """
    if len(front) == 0:
        return
    
    # Resetear distancias
    for ind in front:
        ind.crowding_distance = 0.0
    
    if len(front) == 1:
        front[0].crowding_distance = float('inf')
        return
    
    # Calcular para cada objetivo
    objectives = ['f1', 'f2']
    
    for obj in objectives:
        # Ordenar por objetivo
        front_sorted = sorted(front, key=lambda x: getattr(x, obj))
        
        # Extremos reciben distancia infinita
        front_sorted[0].crowding_distance = float('inf')
        front_sorted[-1].crowding_distance = float('inf')
        
        # Rango del objetivo
        obj_min = getattr(front_sorted[0], obj)
        obj_max = getattr(front_sorted[-1], obj)
        obj_range = obj_max - obj_min
        
        if obj_range == 0:
            continue
        
        # Calcular distancia para individuos intermedios
        for i in range(1, len(front_sorted) - 1):
            prev_obj = getattr(front_sorted[i - 1], obj)
            next_obj = getattr(front_sorted[i + 1], obj)
            front_sorted[i].crowding_distance += (next_obj - prev_obj) / obj_range


def tournament_selection(population: List[Individual], 
                         tournament_size: int = 2) -> Individual:
    """
    Selección por torneo con dominancia de Pareto y crowding distance.
    
    Args:
        population: Población actual
        tournament_size: Tamaño del torneo
    
    Returns:
        Individuo ganador
    """
    # Seleccionar candidatos aleatorios
    candidates = np.random.choice(population, size=tournament_size, replace=False)
    
    # Encontrar el mejor
    best = candidates[0]
    for candidate in candidates[1:]:
        # Preferir menor rank (mejor frente)
        if candidate.rank < best.rank:
            best = candidate
        elif candidate.rank == best.rank:
            # Mismo frente: preferir mayor crowding distance (más diverso)
            if candidate.crowding_distance > best.crowding_distance:
                best = candidate
    
    return best


def crossover(parent1: Individual, parent2: Individual, seed: int = None) -> Individual:
    """
    Operador de cruce: promedia las máscaras y las binariza.
    
    Args:
        parent1: Primer padre
        parent2: Segundo padre
        seed: Semilla aleatoria
    
    Returns:
        Nuevo individuo (offspring)
    """
    if seed is not None:
        set_seed(seed)
    
    offspring_mask = []
    
    for m1, m2 in zip(parent1.mask, parent2.mask):
        # Promediar máscaras
        m_avg = (m1.float() + m2.float()) / 2.0
        
        # Binarizar con umbral 0.5 + ruido pequeño para romper empates
        m_offspring = (m_avg > (0.5 + torch.randn_like(m_avg) * 0.01)).float()
        
        offspring_mask.append(m_offspring)

    # Reparar conectividad del descendiente
    offspring_mask = repair_mask_connectivity(
        offspring_mask, seed=seed if seed is not None else None
    )
    # Garantizar al menos un camino completo input→output
    offspring_mask = ensure_input_output_path(
        offspring_mask, seed=seed if seed is not None else None
    )

    return Individual(offspring_mask)


def mutate(individual: Individual, mutation_rate: float = 0.01, seed: int = None):
    """
    Operador de mutación: flip de bits con probabilidad mutation_rate.
    
    Args:
        individual: Individuo a mutar (se modifica in-place)
        mutation_rate: Probabilidad de mutar cada bit
        seed: Semilla aleatoria
    """
    if seed is not None:
        set_seed(seed)
    
    for mask_layer in individual.mask:
        # Generar máscara de mutación
        mutation_mask = torch.rand_like(mask_layer) < mutation_rate
        
        # Flip bits seleccionados
        mask_layer[mutation_mask] = 1.0 - mask_layer[mutation_mask]

    # Reparar conectividad después de la mutación
    individual.mask = repair_mask_connectivity(
        individual.mask, seed=seed if seed is not None else None
    )
    # Garantizar al menos un camino completo input→output
    individual.mask = ensure_input_output_path(
        individual.mask, seed=seed if seed is not None else None
    )


def tabu_search_improvement(model, individual: Individual, 
                           X_val, y_val,
                           tenure: int = 5,
                           max_iter: int = 20,
                           max_no_improve: int = 10,
                           seed: int = None,
                           progress_callback: Callable[[int, float, float], None] = None,
                           global_eval_counter: List[int] = None) -> Tuple[Individual, int]:
    """
    Mejora local con Tabu Search.
    Usa f1 (1 - accuracy) como criterio principal de mejora.
    
    Args:
        model: Modelo MLP
        individual: Individuo a mejorar
        X_val, y_val: Datos de validación
        tenure: Tamaño de la lista tabú
        max_iter: Máximo de iteraciones
        max_no_improve: Máximo de iteraciones sin mejora
        seed: Semilla aleatoria
        progress_callback: Callback para registrar progreso (eval_count, current_acc, best_acc)
        global_eval_counter: Contador global de evaluaciones [int] (lista mutable)
    
    Returns:
        Tupla (individuo_mejorado, evaluaciones_realizadas)
    """
    if seed is not None:
        set_seed(seed)
    
    tabu_list = TabuList(tenure)
    current = Individual(individual.mask)
    current.evaluate(model, X_val, y_val)
    
    # Actualizar contador global y callback
    if global_eval_counter is not None:
        global_eval_counter[0] += 1
        if progress_callback is not None:
            progress_callback(global_eval_counter[0], current.accuracy, current.accuracy)
    
    best = Individual(current.mask)
    best.accuracy = current.accuracy
    best.sparsity = current.sparsity
    best.f1 = current.f1
    best.f2 = current.f2
    
    evaluations = 1  # Ya evaluamos current
    no_improve_count = 0
    
    for iteration in range(max_iter):
        # Generar vecindario mediante swap_1to1
        neighbors = []
        neighbor_moves = []
        
        for _ in range(20):  # Tamaño de vecindario
            neighbor_mask = [m.clone() for m in current.mask]

            # Aplicar swap_1to1 (la función ya elige internamente el layer a modificar)
            neighbor_mask = swap_1to1(neighbor_mask)

            # Reparar conectividad del vecino generado
            neighbor_mask = repair_mask_connectivity(
                neighbor_mask, seed=seed if seed is not None else None
            )
            # Garantizar al menos un camino completo input→output
            neighbor_mask = ensure_input_output_path(
                neighbor_mask, seed=seed if seed is not None else None
            )
            
            # Crear movimiento (hash de la máscara modificada)
            move = hash(tuple(m.flatten().cpu().numpy().tobytes() for m in neighbor_mask))
            
            neighbors.append(neighbor_mask)
            neighbor_moves.append(move)
        
        # Evaluar vecinos
        best_neighbor = None
        best_neighbor_f1 = float('inf')
        best_neighbor_move = None
        
        for neighbor_mask, move in zip(neighbors, neighbor_moves):
            # Aspiración: aceptar si es mejor que el best global, incluso si es tabú
            neighbor_ind = Individual(neighbor_mask)
            neighbor_ind.evaluate(model, X_val, y_val)
            evaluations += 1
            
            # Actualizar contador global y callback
            if global_eval_counter is not None:
                global_eval_counter[0] += 1
                if progress_callback is not None:
                    progress_callback(global_eval_counter[0], neighbor_ind.accuracy, best.accuracy)
            
            is_tabu = tabu_list.is_tabu(move)
            aspiration = neighbor_ind.f1 < best.f1  # Minimizar f1
            
            if (not is_tabu or aspiration) and neighbor_ind.f1 < best_neighbor_f1:
                best_neighbor = neighbor_ind
                best_neighbor_f1 = neighbor_ind.f1
                best_neighbor_move = move
        
        # Actualizar
        if best_neighbor is not None:
            current = best_neighbor
            tabu_list.add_move(best_neighbor_move)
            
            if current.f1 < best.f1:  # Minimizar f1
                best = Individual(current.mask)
                best.accuracy = current.accuracy
                best.sparsity = current.sparsity
                best.f1 = current.f1
                best.f2 = current.f2
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        tabu_list.update_iteration()
        
        # Criterio de parada
        if no_improve_count >= max_no_improve:
            break
    
    return best, evaluations


def run_ga_tabu_nsga2_experiment(model,
                                 mask_init: List[torch.Tensor],
                                 X_val, y_val,
                                 X_test, y_test,
                                 pop_size: int = 50,
                                 max_generations: int = 100,
                                 mutation_rate: float = 0.01,
                                 crossover_rate: float = 0.9,
                                 tournament_size: int = 2,
                                 min_sparsity: float = 0.1,
                                 max_sparsity: float = 0.9,
                                 tabu_tenure: int = 5,
                                 tabu_max_iter: int = 20,
                                 tabu_max_no_improve: int = 10,
                                 apply_tabu_every: int = 10,
                                 max_evals: int = 10000,
                                 plot_convergence: bool = True,
                                 plot_save_path: str = None,
                                 seed: int = None,
                                 verbose: bool = True) -> Dict:
    """
    Ejecuta GA + Tabu Search + NSGA-II para búsqueda de sub-redes.
    
    Algoritmo GA híbrido multiobjetivo con NSGA-II:
    1. Inicializar población con sparsities variadas
    2. Evaluar población (f1, f2)
    3. Mientras no se cumpla criterio de parada:
        a. Ordenamiento no dominado (calcular frentes de Pareto)
        b. Calcular crowding distance
        c. Selección por torneo (rank + crowding distance)
        d. Crossover + Mutación
        e. Evaluación de offspring
        f. Mejora local con Tabu Search (cada N generaciones sobre frente de Pareto)
        g. Reemplazo elitista (preservar mejores según Pareto)
    
    Objetivos (ambos de minimización):
    - f1 = 1 - accuracy (minimizar error)
    - f2 = sparsity (minimizar conexiones podadas)
    
    Args:
        model: Modelo MLP congelado
        mask_init: Máscara inicial (no usada, solo para compatibilidad)
        X_val, y_val: Datos de validación
        X_test, y_test: Datos de test
        pop_size: Tamaño de la población
        max_generations: Máximo de generaciones
        mutation_rate: Probabilidad de mutación
        crossover_rate: Probabilidad de crossover
        tournament_size: Tamaño del torneo
        min_sparsity: Sparsity mínima a explorar
        max_sparsity: Sparsity máxima a explorar
        tabu_tenure: Tenure para Tabu Search
        tabu_max_iter: Iteraciones máximas de Tabu Search
        tabu_max_no_improve: Iteraciones sin mejora en Tabu Search
        apply_tabu_every: Aplicar Tabu Search cada N generaciones
        max_evals: Máximo de evaluaciones
        plot_convergence: Si es True, genera gráfico de convergencia
        plot_save_path: Ruta para guardar el gráfico (si None, usa 'ga_tabu_nsga2_convergence.png')
        seed: Semilla aleatoria
        verbose: Verbosidad
    
    Returns:
        Dict con resultados
    """
    if seed is not None:
        set_seed(seed)
    
    start_time = time.time()
    
    # Inicializar población
    if verbose:
        print(f"\n{'='*70}")
        print(f"🧬 GA + Tabu Search + NSGA-II (Multiobjetivo)")
        print(f"{'='*70}")
        print(f"Población: {pop_size}")
        print(f"Generaciones: {max_generations}")
        print(f"Sparsity range: [{min_sparsity:.2f}, {max_sparsity:.2f}]")
        print(f"Método: Dominancia de Pareto + Crowding Distance")
        print(f"{'='*70}\n")
    
    mask_shapes = [m.shape for m in mask_init]
    population = initialize_population(mask_shapes, pop_size, min_sparsity, max_sparsity, seed)
    
    # Evaluar población inicial
    total_evaluations = 0
    for ind in population:
        ind.evaluate(model, X_val, y_val)
        total_evaluations += 1
    
    # Tracking
    pareto_front_history = []  # [(gen, [f1s], [f2s], [accs], [sparsities])]
    pareto_snapshots = []      # [(f1_vals, f2_vals)] para plotting
    snapshot_generations = []  # [gen]
    best_accuracy_history = []
    median_f1_history = []
    evaluations_history = []
    
    dense_evaluations = []  # Cada evaluación individual
    dense_best_history = []  # Mejor accuracy global en cada evaluación
    global_eval_counter = [total_evaluations]  # Contador mutable para callbacks
    
    # Mejor individuo global (por f1 - menor error)
    best_individual = min(population, key=lambda x: x.f1)
    
    # Callback para registrar progreso detallado
    def progress_callback(eval_count: int, current_acc: float, best_acc: float):
        """Registra cada evaluación para convergencia detallada."""
        dense_evaluations.append(eval_count)
        dense_best_history.append(best_acc)
    
    # Registrar población inicial
    for _ in range(total_evaluations):
        dense_evaluations.append(len(dense_evaluations) + 1)
        dense_best_history.append(best_individual.accuracy)
    
    generation = 0
    
    # Loop principal
    while generation < max_generations and total_evaluations < max_evals:
        if verbose:
            print(f"\n Generación {generation + 1}/{max_generations}")
        
        # 1. Ordenamiento no dominado
        fronts = fast_non_dominated_sort(population)
        
        # 2. Calcular crowding distance para cada frente
        for front in fronts:
            calculate_crowding_distance(front)
        
        # 3. Tracking del frente de Pareto
        pareto_front = fronts[0]
        pareto_f1s = [ind.f1 for ind in pareto_front]
        pareto_f2s = [ind.f2 for ind in pareto_front]
        pareto_accs = [ind.accuracy for ind in pareto_front]
        pareto_sparsities = [ind.sparsity for ind in pareto_front]
        pareto_front_history.append((generation, pareto_f1s.copy(), pareto_f2s.copy(), 
                         pareto_accs.copy(), pareto_sparsities.copy()))

        # Guardar snapshot para graficar convergencia (frente de Pareto en esta generación)
        pareto_snapshots.append((pareto_f1s.copy(), pareto_f2s.copy()))
        snapshot_generations.append(generation)
        
        # 4. Tracking de métricas
        all_f1 = [ind.f1 for ind in population]
        all_accs = [ind.accuracy for ind in population]
        
        best_accuracy_history.append(max(all_accs))
        median_f1_history.append(np.median(all_f1))
        evaluations_history.append(total_evaluations)
        
        if verbose:
            print(f"   Frente de Pareto: {len(pareto_front)} individuos")
            print(f"   Mejor accuracy: {max(pareto_accs)*100:.2f}%")
            print(f"   Mejor sparsity: {min(pareto_sparsities):.3f}")
            print(f"   Rango f1: [{min(pareto_f1s):.4f}, {max(pareto_f1s):.4f}]")
            print(f"   Evaluaciones: {total_evaluations}/{max_evals}")
        
        # 5. Actualizar mejor global
        current_best = min(population, key=lambda x: x.f1)
        if current_best.f1 < best_individual.f1:
            best_individual = Individual(current_best.mask)
            best_individual.accuracy = current_best.accuracy
            best_individual.sparsity = current_best.sparsity
            best_individual.f1 = current_best.f1
            best_individual.f2 = current_best.f2
        
        # 6. Generar offspring
        offspring = []
        
        while len(offspring) < pop_size:
            # Selección por torneo
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            
            # Crossover
            if np.random.rand() < crossover_rate:
                child = crossover(parent1, parent2, seed=seed + generation if seed else None)
            else:
                # Sin crossover, clonar un padre
                child = Individual(parent1.mask)
            
            # Mutación
            if np.random.rand() < mutation_rate:
                mutate(child, mutation_rate, seed=seed + generation if seed else None)
            
            offspring.append(child)
        
        # 7. Evaluar offspring
        for child in offspring:
            child.evaluate(model, X_val, y_val)
            total_evaluations += 1
            global_eval_counter[0] += 1
            
            # Actualizar mejor global
            if child.f1 < best_individual.f1:
                best_individual = Individual(child.mask)
                best_individual.accuracy = child.accuracy
                best_individual.sparsity = child.sparsity
                best_individual.f1 = child.f1
                best_individual.f2 = child.f2
            
            # Registrar convergencia detallada
            dense_evaluations.append(global_eval_counter[0])
            dense_best_history.append(best_individual.accuracy)
            
            if total_evaluations >= max_evals:
                break
        
        # 8. Mejora local con Tabu Search (cada N generaciones sobre frente de Pareto)
        if (generation + 1) % apply_tabu_every == 0 and total_evaluations < max_evals:
            if verbose:
                print(f"    Aplicando Tabu Search al frente de Pareto...")
            
            improved_front = []
            for ind in pareto_front[:min(5, len(pareto_front))]:  # Solo los mejores 5
                if total_evaluations >= max_evals:
                    break
                
                improved_ind, evals = tabu_search_improvement(
                    model, ind, X_val, y_val,
                    tenure=tabu_tenure,
                    max_iter=tabu_max_iter,
                    max_no_improve=tabu_max_no_improve,
                    seed=seed + generation if seed else None,
                    progress_callback=progress_callback,
                    global_eval_counter=global_eval_counter
                )
                
                total_evaluations += evals
                
                # Actualizar mejor global si Tabu encontró algo mejor
                if improved_ind.f1 < best_individual.f1:
                    best_individual = Individual(improved_ind.mask)
                    best_individual.accuracy = improved_ind.accuracy
                    best_individual.sparsity = improved_ind.sparsity
                    best_individual.f1 = improved_ind.f1
                    best_individual.f2 = improved_ind.f2
                
                improved_front.append(improved_ind)
            
            # Agregar individuos mejorados a la población
            population.extend(improved_front)
        
        # 9. Reemplazo elitista: combinar población actual + offspring
        combined = population + offspring
        
        # Re-ordenar con dominancia de Pareto
        fronts = fast_non_dominated_sort(combined)
        
        # Calcular crowding distance
        for front in fronts:
            calculate_crowding_distance(front)
        
        # Seleccionar mejores pop_size individuos
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= pop_size:
                new_population.extend(front)
            else:
                # Ordenar por crowding distance y tomar los mejores
                front_sorted = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
                new_population.extend(front_sorted[:pop_size - len(new_population)])
                break
        
        population = new_population
        generation += 1
    
    elapsed_time = time.time() - start_time
    
    # Evaluación final en test
    best_individual.accuracy = score(model, best_individual.mask, X_val, y_val, metric='accuracy')
    acc_test = score(model, best_individual.mask, X_test, y_test, metric='accuracy')

    # Frente de Pareto final (no dominado) de la última generación
    final_fronts = fast_non_dominated_sort(population)
    final_pareto_front = final_fronts[0]
    
    if verbose:
        print(f"\n{'='*70}")
        print(f" GA + Tabu Search + NSGA-II completado")
        print(f"{'='*70}")
        print(f"Mejor individuo (menor f1):")
        print(f"  f1 (1-acc): {best_individual.f1:.4f}")
        print(f"  f2 (sparsity): {best_individual.f2:.4f}")
        print(f"Accuracy (val): {best_individual.accuracy*100:.2f}%")
        print(f"Accuracy (test): {acc_test*100:.2f}%")
        print(f"Sparsity: {best_individual.sparsity:.3f}")
        print(f"Frente de Pareto final: {len(final_pareto_front)} individuos")
        print(f"Evaluaciones totales: {total_evaluations}")
        print(f"Tiempo: {elapsed_time:.2f}s")
        print(f"{'='*70}\n")

    # Plot de convergencia del frente de Pareto (opcional)
    if plot_convergence and len(pareto_snapshots) > 0:
        def hypervolume_2d_min(points: List[Dict[str, float]], ref: Tuple[float, float] = (1.0, 1.0)) -> float:
            """Hipervolumen 2D para problemas de minimización (f1,f2 >=0).

            Aproximación por suma de áreas rectangulares desde cada punto al punto de referencia.
            Asume que los puntos representan un frente no dominado.
            """
            # Ordenar por f1 ascendente
            pts = sorted(points, key=lambda p: p['f1'])
            hv = 0.0
            prev_f1 = ref[0]
            for p in reversed(pts):
                width = prev_f1 - p['f1']
                height = ref[1] - p['f2']
                if width > 0 and height > 0:
                    hv += width * height
                prev_f1 = p['f1']
            return hv

        def plot_pareto_convergence(pareto_snapshots_local,
                        snapshot_generations_local,
                        save_path=None,
                        figsize=(10, 8),
                        ref=(1.0, 1.0)) -> None:
            """Grafica la convergencia del frente de Pareto en múltiples generaciones.

            Args:
                pareto_snapshots_local: lista de tuplas (f1_vals, f2_vals) por generación
                snapshot_generations_local: lista de números de generación
                save_path: ruta donde guardar el gráfico
                figsize: tamaño de la figura
                ref: punto de referencia para hipervolumen
            """
            if len(pareto_snapshots_local) == 0:
                print("No hay snapshots para graficar")
                return

            plt.figure(figsize=figsize)
            # Colores: última gen (azul) <- amarillo <- naranja <- primera (rojo)
            colors = ['darkred', 'orangered', 'darkorange', 'blue']

            for idx, (f1_vals, f2_vals) in enumerate(pareto_snapshots_local):
                gen = snapshot_generations_local[idx]
                color = colors[idx % len(colors)]

                # Hipervolumen
                hv_points = [{'f1': f1, 'f2': f2} for f1, f2 in zip(f1_vals, f2_vals)]
                hv = hypervolume_2d_min(hv_points, ref=ref)

                # Ordenar por f1 para conectar con línea punteada
                points = sorted(zip(f1_vals, f2_vals), key=lambda p: p[0])
                sorted_f1 = [p[0] for p in points]
                sorted_f2 = [p[1] for p in points]

                plt.plot(sorted_f1, sorted_f2, color=color,
                         linestyle='--', linewidth=1.5, alpha=0.5)
                plt.scatter(f1_vals, f2_vals, c=color, alpha=0.6, s=50,
                            label=f'Gen {gen} (HV={hv:.4f})')

            plt.xlabel('f1 (1 - accuracy)', fontsize=12)
            plt.ylabel('f2 (sparsity)', fontsize=12)
            plt.title('Convergencia del Frente de Pareto en NSGA-II', fontsize=14)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Guardar gráfico en lugar de mostrarlo
            if save_path is None:
                save_path = 'ga_tabu_nsga2_convergence.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            if verbose:
                print(f"   Gráfico de convergencia guardado en: {save_path}")

        # Seleccionar sólo 4 generaciones equidistantes para el gráfico
        if len(snapshot_generations) >= 4:
            first_gen = snapshot_generations[0]
            last_gen = snapshot_generations[-1]

            # 4 índices equidistantes entre la primera y la última generación
            target_gens = np.linspace(first_gen, last_gen, 4, dtype=int)
            target_gens = sorted(set(target_gens))  # evitar duplicados por redondeo

            gen_to_snapshot = {g: s for g, s in zip(snapshot_generations, pareto_snapshots)}
            filtered_generations = []
            filtered_snapshots = []

            for g in target_gens:
                # Si alguna generación exacta falta, usar la más cercana disponible
                if g not in gen_to_snapshot:
                    closest = min(snapshot_generations, key=lambda x: abs(x - g))
                    g = closest
                if g in gen_to_snapshot and g not in filtered_generations:
                    filtered_generations.append(g)
                    filtered_snapshots.append(gen_to_snapshot[g])
        else:
            # Si hay menos de 4 generaciones, usar todas
            filtered_generations = snapshot_generations
            filtered_snapshots = pareto_snapshots

        plot_pareto_convergence(filtered_snapshots, filtered_generations, save_path=plot_save_path)

    # Calcular hipervolumen del frente final reutilizando la implementación anterior
    final_hv_points = [{'f1': ind.f1, 'f2': ind.f2} for ind in final_pareto_front]
    final_hv = 0.0
    if len(final_hv_points) > 0:
        pts = sorted(final_hv_points, key=lambda p: p['f1'])
        prev_f1 = 1.0
        for p in reversed(pts):
            width = prev_f1 - p['f1']
            height = 1.0 - p['f2']
            if width > 0 and height > 0:
                final_hv += width * height
            prev_f1 = p['f1']
    
    # Compilar resultados
    results = {
        'best_mask': best_individual.mask,
        'acc_explore': best_individual.accuracy,
        'acc_test': acc_test,
        'final_pareto_front': final_pareto_front,
        'final_hv': final_hv,
        'seed_search': seed,
        'hyperparameters': {
            'pop_size': pop_size,
            'max_generations': max_generations,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'tournament_size': tournament_size,
            'min_sparsity': min_sparsity,
            'max_sparsity': max_sparsity,
            'tabu_tenure': tabu_tenure,
            'tabu_max_iter': tabu_max_iter,
            'tabu_max_no_improve': tabu_max_no_improve,
            'apply_tabu_every': apply_tabu_every,
        },
        'trace': {
            'evaluations': evaluations_history,
            'best_accuracy_history': best_accuracy_history,
            'median_f1_history': median_f1_history,
            'pareto_front_history': pareto_front_history,
            'dense_evaluations': dense_evaluations,
            'dense_global_best_history': dense_best_history,
            'final_stats': {
                'total_evaluations': total_evaluations,
                'total_time': elapsed_time,
                'active_connections': count_active_connections(best_individual.mask),
                'final_sparsity': best_individual.sparsity,
                'final_f1': best_individual.f1,
                'final_f2': best_individual.f2,
                'pareto_front_size': len(pareto_front),
                'hypervolume': final_hv,  # Hipervolumen del frente de Pareto final
            }
        }
    }
    
    return results
