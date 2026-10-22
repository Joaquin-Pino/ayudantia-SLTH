"""
Orquestador principal de experimentos SLTH.
Ejecuta las configuraciones en base al config definido
"""

import yaml
import os
import sys
import numpy as np
import pandas as pd
from itertools import product
import torch
from typing import Dict, List, Tuple
import traceback

# Imports locales
from seeds import set_seed
from model import build_mlp
from mask import create_mask, create_magnitude_mask
from eval import score
from data_loader import load_wine_data, load_dataset, get_data_info
from gls import run_gls_experiment
from scatter_search import run_ss_experiment
from ga_tabu_nsga2 import run_ga_tabu_nsga2_experiment
from utils import save_run_to_csv, ensure_dir, Timer, generate_report
from logging_detailed import ObjectiveFunctionLogger, StrategyResultsLogger
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(config_path: str) -> Dict:
    """Carga archivo de configuración YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_testbed_instances(config: Dict) -> List[Dict]:
    """
    Genera todas las instancias del testbed combinando:
    - arquitecturas
    - sparsities
    - seed_model
    - seed_search
    
    Returns:
        Lista de instancias (cada una es un Dict con la configuración)
    """
    architectures = config['architectures']
    sparsities = config['sparsities']
    seed_models = config['seed_model']
    seed_searches = config['seed_search']
    
    instances = []
    
    for arch, sparsity, seed_model, seed_search in product(
        architectures, sparsities, seed_models, seed_searches
    ):
        instance = {
            'architecture': arch,
            'sparsity': sparsity,
            'seed_model': seed_model,
            'seed_search': seed_search,
            'method': config['method']
        }
        instances.append(instance)
    
    return instances


def run_random_baseline(model, X_val, y_val, X_test, y_test, 
                       sparsity: float, num_repetitions: int, seed_base: int = 1000) -> Dict:
    """
    Ejecuta baseline de máscaras aleatorias.
    
    Returns:
        Dict con estadísticas (mean, median, std de acc_explore y acc_test)
    """
    shapes = model.get_weight_shapes()
    
    acc_explores = []
    acc_tests = []
    
    for i in range(num_repetitions):
        mask = create_mask(shapes, sparsity=sparsity, per_layer=True, seed=seed_base + i)
        acc_explore = score(model, mask, X_val, y_val)
        acc_test = score(model, mask, X_test, y_test)
        acc_explores.append(acc_explore)
        acc_tests.append(acc_test)
    
    return {
        'acc_explore_mean': np.mean(acc_explores),
        'acc_explore_median': np.median(acc_explores),
        'acc_explore_std': np.std(acc_explores),
        'acc_test_mean': np.mean(acc_tests),
        'acc_test_median': np.median(acc_tests),
        'acc_test_std': np.std(acc_tests)
    }


def run_magnitude_baseline(model, X_val, y_val, X_test, y_test, 
                           sparsity: float, local_search_iters: int = 0) -> Dict:
    """
    Ejecuta baseline de máscara por magnitud (con opcional búsqueda local).
    
    Returns:
        Dict con acc_explore y acc_test
    """
    mask = create_magnitude_mask(model, sparsity=sparsity, per_layer=True)
    
    # Opcional: búsqueda local first-improvement
    if local_search_iters > 0:
        from mask import swap_1to1
        current_acc = score(model, mask, X_val, y_val)
        
        for _ in range(local_search_iters):
            neighbor = swap_1to1(mask)
            neighbor_acc = score(model, neighbor, X_val, y_val)
            
            if neighbor_acc > current_acc:
                mask = neighbor
                current_acc = neighbor_acc
    
    acc_explore = score(model, mask, X_val, y_val)
    acc_test = score(model, mask, X_test, y_test)
    
    return {
        'acc_explore': acc_explore,
        'acc_test': acc_test
    }

def run_single_experiment(instance: Dict, config: Dict, 
                         X_val, y_val, X_test, y_test, 
                         verbose: bool = False,
                         obj_logger = None,
                         strategy_logger = None,
                         replica_id: int = 0) -> Dict:
    """
    Ejecuta un experimento individual (una instancia del testbed).
    
    Args:
        instance: Configuración de la instancia
        config: Configuración global
        X_val, y_val, X_test, y_test: Datos
        verbose: Verbosidad
        obj_logger: Logger de llamadas a función objetivo (opcional)
        strategy_logger: Logger de resultados de estrategias (opcional)
        replica_id: ID de la réplica (0-10 para 11 réplicas)
    
    Returns:
        Dict con resultados
    """
    # Helper para representar la arquitectura en nombres de archivos
    def arch_signature(arch: List[int]) -> str:
        # Ej: [13, 128, 128, 3] -> "13-128-128-3"
        return "-".join(str(x) for x in arch)

    # Crear modelo con seed_model
    model = build_mlp(instance['architecture'], seed_model=instance['seed_model'])
    
    # Crear máscara inicial
    shapes = model.get_weight_shapes()
    mask_init = create_mask(shapes, sparsity=instance['sparsity'], 
                           per_layer=True, seed=instance['seed_search'])
    
    # Seleccionar método
    method = instance['method']
    
    # Iniciar logging de estrategia
    if strategy_logger is not None:
        strategy_logger.start_strategy(
            strategy_code=method,
            seed_model=instance['seed_model'],
            seed_search=instance['seed_search']
        )
    
    if method == "GLS":
        hp_method = config['GLS']
        results = run_gls_experiment(
            model=model,
            mask_init=mask_init,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            hp_gls=hp_method,
            seed_search=instance['seed_search'],
            verbose=verbose,
            obj_logger=obj_logger
        )
    
    elif method == "SS":
        hp_method = config['SS']
        results = run_ss_experiment(
            model=model,
            mask_init=mask_init,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            hp_ss=hp_method,
            seed_search=instance['seed_search'],
            verbose=verbose,
            obj_logger=obj_logger
        )
    
    elif method == "GA_TABU_NSGA2":
        hp_method = config['GA_TABU_NSGA2']
        plot_convergence = config['logging'].get('plot_convergence', True)

        plot_save_path = None
        if plot_convergence:
            arch_sig = arch_signature(instance['architecture'])
            plot_filename = (
                f"pareto_conv_arch_{arch_sig}_"
                f"sp{instance['sparsity']}_"
                f"sm{instance['seed_model']}_"
                f"ss{instance['seed_search']}_"
                f"rep{replica_id}.png"
            )
            plot_save_path = os.path.join(config['output']['plots_dir'], plot_filename)
            ensure_dir(config['output']['plots_dir'])

        results = run_ga_tabu_nsga2_experiment(
            model=model,
            mask_init=mask_init,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            pop_size=hp_method['pop_size'],
            max_generations=hp_method['max_generations'],
            mutation_rate=hp_method['mutation_rate'],
            crossover_rate=hp_method['crossover_rate'],
            tournament_size=hp_method['tournament_size'],
            min_sparsity=hp_method['min_sparsity'],
            max_sparsity=hp_method['max_sparsity'],
            tabu_tenure=hp_method['tabu_tenure'],
            tabu_max_iter=hp_method['tabu_max_iter'],
            tabu_max_no_improve=hp_method['tabu_max_no_improve'],
            apply_tabu_every=hp_method['apply_tabu_every'],
            max_evals=hp_method['max_evals'],
            plot_convergence=plot_convergence,
            plot_save_path=plot_save_path,
            seed=instance['seed_search'],
            verbose=verbose
        )
    
    else:
        raise ValueError(f"Método no reconocido: {method}")
    
    # Capturar info de memoria y solución para runs.csv
    memory_mb = 0
    solution_repr = ""
    if strategy_logger is not None:
        # Finalizar logging de estrategia
        strategy_logger.end_strategy(
            best_mask=results['best_mask'],
            best_fitness=results['acc_explore']
        )
        # Obtener info del último resultado
        if len(strategy_logger.results) > 0:
            last_result = strategy_logger.results[-1]
            memory_mb = last_result.get('memoria_ejecucion_mb', 0)
            solution_repr = last_result.get('solution_repr', "")
    
    # Compilar resultados
    # Obtener hipervolumen: puede estar en results['final_hv'] (GA) o trace['final_stats']['hypervolume'] (GLS, SS)
    hypervolume = results.get('final_hv', None) or results['trace']['final_stats'].get('hypervolume', None)
    
    run_data = {
        'architecture': str(instance['architecture']),
        'sparsity': instance['sparsity'],
        'seed_model': instance['seed_model'],
        'seed_search': instance['seed_search'],
        'replica_id': replica_id,
        'method': method,
        'acc_explore': results['acc_explore'],
        'acc_test': results['acc_test'],
        'evaluations': results['trace']['final_stats']['total_evaluations'],
        'time_seconds': results['trace']['final_stats']['total_time'],
        'active_connections': results['trace']['final_stats']['active_connections'],
        'memory_mb': memory_mb,
        'solution_repr': solution_repr,
        'hypervolume': hypervolume,  # Hipervolumen unificado (GA, GLS, SS)
        'final_hv': results.get('final_hv', None),  # Mantener por compatibilidad
        **results['hyperparameters']
    }
    
    # Guardar artefactos
    output_config = config['output']
    arch_sig = arch_signature(instance['architecture'])
    save_masks = config['logging'].get('save_masks', True)

    mask_filename = (
        f"mask_arch_{arch_sig}_"
        f"sp{instance['sparsity']}_"
        f"sm{instance['seed_model']}_"
        f"ss{instance['seed_search']}_"
        f"rep{replica_id}.pt"
    )

    if save_masks:
        mask_path = os.path.join(output_config['masks_dir'], mask_filename)
        ensure_dir(os.path.dirname(mask_path))
        torch.save({
            'best_mask': results['best_mask'],
            'trace': results['trace'],
            'acc_explore': results['acc_explore'],
            'acc_test': results['acc_test'],
            'hyperparameters': results['hyperparameters'],
            'seed_search': results['seed_search'],
            'architecture': instance['architecture'],
            'sparsity': instance['sparsity'],
            'seed_model': instance['seed_model'],
            'method': instance['method']
        }, mask_path)
        run_data['mask_path'] = mask_path
    else:
        run_data['mask_path'] = ""

    # Si es GA_TABU_NSGA2, guardar también el frente de Pareto final completo
    if save_masks and method == "GA_TABU_NSGA2" and 'final_pareto_front' in results:
        from mask import count_active_connections as _count_ac
        import csv as _csv

        pareto_dir = os.path.join(output_config['masks_dir'], "pareto_fronts")
        ensure_dir(pareto_dir)

        # Prefijo único por ejecución (coincide con mask_filename)
        base_name = os.path.splitext(os.path.basename(mask_filename))[0]

        # 1) Guardar todas las máscaras del frente final como .pt individuales
        for idx, ind in enumerate(results['final_pareto_front']):
            pareto_mask_filename = f"{base_name}_pareto_{idx}.pt"
            pareto_mask_path = os.path.join(pareto_dir, pareto_mask_filename)
            torch.save({
                'mask': ind.mask,
                'f1': ind.f1,
                'f2': ind.f2,
                'accuracy_val': ind.accuracy,
                'sparsity': ind.sparsity,
                'architecture': instance['architecture'],
                'sparsity_init': instance['sparsity'],
                'seed_model': instance['seed_model'],
                'seed_search': instance['seed_search'],
                'replica_id': replica_id,
            }, pareto_mask_path)

        # 2) CSV por frente final (una fila por solución del frente)
        pareto_csv_path = os.path.join(
            pareto_dir,
            f"pareto_front_{base_name}.csv",
        )

        header = [
            'architecture',
            'sparsity_init',
            'seed_model',
            'seed_search',
            'replica_id',
            'pareto_index',
            'acc_explore',
            'acc_test',
            'sparsity',
            'f1',
            'f2',
            'active_connections',
            'final_hv',
            'mask_file',
        ]

        file_exists = os.path.isfile(pareto_csv_path)
        with open(pareto_csv_path, 'a', newline='') as f:
            writer = _csv.writer(f)
            if not file_exists:
                writer.writerow(header)

            for idx, ind in enumerate(results['final_pareto_front']):
                # accuracy de exploración ya está en ind.accuracy; acc_test sólo para el mejor
                acc_explore = ind.accuracy
                acc_test = None
                # usamos acc_test del mejor individuo sólo para la fila cuyo mask coincide
                if torch.allclose(torch.cat([m.flatten() for m in ind.mask]),
                                  torch.cat([m.flatten() for m in results['best_mask']])):
                    acc_test = results['acc_test']

                sparsity = ind.sparsity
                active_conns = _count_ac(ind.mask)
                mask_file = os.path.join("pareto_fronts", f"{base_name}_pareto_{idx}.pt")

                writer.writerow([
                    str(instance['architecture']),
                    instance['sparsity'],
                    instance['seed_model'],
                    instance['seed_search'],
                    replica_id,
                    idx,
                    acc_explore,
                    acc_test,
                    sparsity,
                    ind.f1,
                    ind.f2,
                    active_conns,
                    results.get('final_hv', None),
                    mask_file,
                ])
    
    # Guardar plot de convergencia (DESACTIVADO - se puede generar después si se necesita)
    # if config['logging']['plot_convergence']:
    #     plot_filename = (
    #         f"conv_arch_{arch_sig}_"
    #         f"sp{instance['sparsity']}_"
    #         f"sm{instance['seed_model']}_"
    #         f"ss{instance['seed_search']}.png"
    #     )
    #     plot_path = os.path.join(output_config['plots_dir'], plot_filename)
    #     
    #     title = f"Arch={instance['architecture']}, ρ={instance['sparsity']}, " \
    #             f"seed_model={instance['seed_model']}, seed_search={instance['seed_search']}"
    #     save_trace_plot(results['trace'], results['acc_test'], plot_path, title=title)
    
    # Visualizaciones (DESACTIVADO - generar después con generate_visualizations_from_masks.py)
    run_data['winner_diagram_path'] = ""
    run_data['mask_summary_path'] = ""
    
    return run_data


def run_instance_with_replicas(instance: Dict, config: Dict,
                               X_val, y_val, X_test, y_test,
                               num_replicas: int = 11,
                               verbose: bool = False,
                               obj_logger = None,
                               strategy_logger = None) -> Tuple[List[Dict], Dict]:
    """
    Ejecuta una instancia múltiples veces (réplicas) para obtener estadísticas robustas.
    
    Args:
        instance: Configuración de la instancia
        config: Configuración global
        X_val, y_val, X_test, y_test: Datos
        num_replicas: Número de réplicas a ejecutar (default: 11)
        verbose: Verbosidad
        obj_logger: Logger de llamadas a función objetivo
        strategy_logger: Logger de resultados de estrategias
    
    Returns:
        Tupla (lista_resultados_individuales, estadísticas_agregadas)
    """
    replicas_results = []
    base_seed_search = instance['seed_search']
    
    print(f"\n{'─'*70}")
    print(f"Ejecutando {num_replicas} replicas de la configuracion:")
    print(f"   Arquitectura: {instance['architecture']}")
    print(f"   Sparsity: {instance['sparsity']}")
    print(f"   Seed model: {instance['seed_model']}")
    print(f"{'─'*70}")
    
    for replica_id in range(num_replicas):
        # Modificar seed de búsqueda para cada réplica
        instance_replica = instance.copy()
        instance_replica['seed_search'] = base_seed_search + replica_id * 1000
        
        print(f"\n  Réplica {replica_id + 1}/{num_replicas} (seed_search={instance_replica['seed_search']})...", end=" ")
        
        try:
            run_data = run_single_experiment(
                instance=instance_replica,
                config=config,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                verbose=verbose,
                obj_logger=obj_logger,
                strategy_logger=strategy_logger,
                replica_id=replica_id
            )
            
            replicas_results.append(run_data)
            print(f"Acc_val={run_data['acc_explore']*100:.2f}%, Acc_test={run_data['acc_test']*100:.2f}%")

        except Exception as e:
            print(f"ERROR: {str(e)}")
            traceback.print_exc()
            continue
    
    # Calcular estadísticas agregadas
    if len(replicas_results) == 0:
        print(f"\n[!] No se completó ninguna réplica exitosamente")
        return replicas_results, {}
    
    # Extraer métricas
    acc_explores = [r['acc_explore'] for r in replicas_results]
    acc_tests = [r['acc_test'] for r in replicas_results]
    evaluations = [r['evaluations'] for r in replicas_results]
    times = [r['time_seconds'] for r in replicas_results]
    
    aggregated_stats = {
        'architecture': str(instance['architecture']),
        'sparsity': instance['sparsity'],
        'seed_model': instance['seed_model'],
        'method': instance['method'],
        'num_replicas': len(replicas_results),
        
        # Estadísticas de accuracy validación
        'acc_explore_mean': np.mean(acc_explores),
        'acc_explore_median': np.median(acc_explores),
        'acc_explore_std': np.std(acc_explores),
        'acc_explore_min': np.min(acc_explores),
        'acc_explore_max': np.max(acc_explores),
        
        # Estadísticas de accuracy test
        'acc_test_mean': np.mean(acc_tests),
        'acc_test_median': np.median(acc_tests),
        'acc_test_std': np.std(acc_tests),
        'acc_test_min': np.min(acc_tests),
        'acc_test_max': np.max(acc_tests),
        
        # Estadísticas de evaluaciones
        'evaluations_mean': np.mean(evaluations),
        'evaluations_median': np.median(evaluations),
        'evaluations_std': np.std(evaluations),
        
        # Estadísticas de tiempo
        'time_mean': np.mean(times),
        'time_median': np.median(times),
        'time_std': np.std(times),
        'time_total': np.sum(times),

        # Hipervolumen medio del frente final (si existe en los resultados individuales)
        'final_hv_mean': np.mean([r['final_hv'] for r in replicas_results
                                  if r.get('final_hv') is not None]) if any(
            r.get('final_hv') is not None for r in replicas_results
        ) else None,
        
        # Hiperparámetros (del primer resultado)
        **{k: v for k, v in replicas_results[0].items() 
           if k in ['alpha', 'lambda', 'max_iter', 'max_no_improve', 'ls_max_neighbors', 'ls_max_iters']}
    }
    
    print(f"\n{'─'*70}")
    print(f"ESTADÍSTICAS AGREGADAS ({len(replicas_results)} réplicas):")
    print(f"   Acc_val:  {aggregated_stats['acc_explore_mean']*100:.2f}% ± {aggregated_stats['acc_explore_std']*100:.2f}% "
          f"[{aggregated_stats['acc_explore_min']*100:.2f}% - {aggregated_stats['acc_explore_max']*100:.2f}%]")
    print(f"   Acc_test: {aggregated_stats['acc_test_mean']*100:.2f}% ± {aggregated_stats['acc_test_std']*100:.2f}% "
          f"[{aggregated_stats['acc_test_min']*100:.2f}% - {aggregated_stats['acc_test_max']*100:.2f}%]")
    print(f"   Tiempo total: {aggregated_stats['time_total']:.1f}s")
    print(f"{'─'*70}\n")
    
    return replicas_results, aggregated_stats


def plot_replicas_analysis(replicas_results: List[Dict], 
                           aggregated_stats: Dict,
                           output_dir: str,
                           config_signature: str):
    """
    Genera gráfico combinado: convergencia (izquierda) + boxplot (derecha).
    
    Args:
        replicas_results: Lista de resultados individuales
        aggregated_stats: Estadísticas agregadas
        output_dir: Directorio donde guardar los plots
        config_signature: Firma de la configuración para nombrar archivos
    """
    if len(replicas_results) == 0:
        return
    
    ensure_dir(output_dir)
    
    # Encontrar la réplica con mejor accuracy final
    best_replica_idx = np.argmax([r['acc_explore'] for r in replicas_results])
    best_replica = replicas_results[best_replica_idx]
    
    # Cargar trace de la mejor réplica
    try:
        mask_data = torch.load(best_replica['mask_path'], weights_only=False)
        trace = mask_data['trace']
        acc_test = mask_data['acc_test']
    except Exception as e:
        print(f"   [!] Error al cargar trace: {str(e)}")
        return
    
    # ═══════════════════════════════════════════════════════════
    # Crear figura con 2 paneles: Convergencia + Boxplot
    # ═══════════════════════════════════════════════════════════
    
    fig = plt.figure(figsize=(18, 6))
    
    # ───────────────────────────────────────────────────────────
    # PANEL IZQUIERDO: Gráfico de Convergencia (accuracy vs evaluaciones)
    # ───────────────────────────────────────────────────────────
    ax_conv = plt.subplot(1, 2, 1)
    
    evals = trace['evaluations']
    
    # Determinar si es SA, GLS, SS o GA_TABU_NSGA2
    if 'current_history' in trace:
        # SA
        ax_conv.plot(evals, [acc*100 for acc in trace['current_history']], 
                    label='Current', alpha=0.6, linewidth=1, color='steelblue')
        ax_conv.plot(evals, [acc*100 for acc in trace['best_history']], 
                    label='Best', linewidth=2, color='darkblue')
    elif 'accuracy_history' in trace:
        # GLS
        ax_conv.plot(trace['evaluations'], [a*100 for a in trace['accuracy_history']], 
                'b-', alpha=0.5, label='Current')
        ax_conv.plot(trace['evaluations'], [a*100 for a in trace['best_accuracy_history']], 
                'r-', linewidth=2, label='Best')
    elif 'refset_best_history' in trace:
        # SS (Scatter Search)
        # Si existe historial denso por evaluación, dibujarlo primero
        if 'dense_evaluations' in trace and len(trace.get('dense_evaluations', [])) > 0:
            ax_conv.plot(
                trace['dense_evaluations'],
                [a*100 for a in trace['dense_global_best_history']],
                color='crimson', alpha=0.35, linewidth=1.5,
                label='Global Best (denso)'
            )
        ax_conv.plot(
            trace['evaluations'], [a*100 for a in trace['refset_mean_history']],
            'g-', alpha=0.5, linewidth=1.5, label='RefSet Mean', marker='o'
        )
        ax_conv.plot(
            trace['evaluations'], [a*100 for a in trace['refset_best_history']],
            'b-', alpha=0.7, linewidth=2, label='RefSet Best', marker='s'
        )
        ax_conv.plot(
            trace['evaluations'], [a*100 for a in trace['best_history']],
            'r-', linewidth=2.5, label='Global Best', marker='D'
        )
    elif 'pareto_front_history' in trace:
        # GA_TABU_NSGA2
        # Si existe historial denso por evaluación, dibujarlo primero (incluye Tabu Search)
        if 'dense_evaluations' in trace and len(trace.get('dense_evaluations', [])) > 0:
            ax_conv.plot(
                trace['dense_evaluations'],
                [a*100 for a in trace['dense_global_best_history']],
                color='crimson', alpha=0.35, linewidth=1.5,
                label='Global Best (con Tabu)'
            )
        # Mostrar evolución por generación
        if 'best_accuracy_history' in trace and len(trace['best_accuracy_history']) > 0:
            ax_conv.plot(
                trace['evaluations'], [a*100 for a in trace['best_accuracy_history']],
                'b-', alpha=0.7, linewidth=2, label='Best por generación', marker='s'
            )
    
    ax_conv.set_xlabel('Evaluaciones', fontsize=11)
    ax_conv.set_ylabel('Accuracy (%)', fontsize=11)
    ax_conv.set_title(f'Convergencia - Mejor Réplica (#{best_replica_idx})\n'
                     f'{aggregated_stats["architecture"]}, ρ={aggregated_stats["sparsity"]}',
                     fontsize=12, fontweight='bold')
    ax_conv.grid(True, alpha=0.3)
    
    # Marcar accuracy en test
    ax_conv.scatter([evals[-1]], [acc_test * 100], s=100, color='crimson', 
                   marker='o', zorder=5, edgecolors='black', linewidth=1.5,
                   label=f'Test: {acc_test*100:.2f}%')
    
    ax_conv.legend(loc='best', fontsize=10)
    
    # ───────────────────────────────────────────────────────────
    # PANEL DERECHO: Boxplot de Distribución de Accuracies
    # ───────────────────────────────────────────────────────────
    ax_box = plt.subplot(1, 2, 2)
    
    acc_explores = [r['acc_explore'] * 100 for r in replicas_results]
    acc_tests = [r['acc_test'] * 100 for r in replicas_results]
    
    # Verificar si hay información de Tabu Search (SS o GA_TABU_NSGA2)
    has_tabu_info = False
    tabu_all_accs = []  # TODAS las accuracies del Tabu (distribución completa)
    method_name = best_replica.get('method', 'Unknown')
    
    if 'refset_best_history' in trace:  # Es Scatter Search
        # Extraer TODAS las accuracies registradas durante el Tabu Search
        # Similar a como GLS muestra accuracy_history (todos los óptimos locales)
        if 'dense_evaluations' in trace and len(trace.get('dense_evaluations', [])) > 0:
            has_tabu_info = True
            # Recolectar TODAS las evaluaciones del Tabu de TODAS las réplicas
            for replica in replicas_results:
                try:
                    replica_mask_data = torch.load(replica['mask_path'], weights_only=False)
                    replica_trace = replica_mask_data['trace']
                    if 'dense_global_best_history' in replica_trace and len(replica_trace['dense_global_best_history']) > 0:
                        # Tomar TODAS las accuracies del historial denso (no solo el máximo)
                        all_tabu_accs = [acc * 100 for acc in replica_trace['dense_global_best_history']]
                        tabu_all_accs.extend(all_tabu_accs)
                except:
                    pass
    elif 'pareto_front_history' in trace:  # Es GA_TABU_NSGA2
        # Extraer TODAS las accuracies registradas (incluyendo Tabu Search)
        if 'dense_evaluations' in trace and len(trace.get('dense_evaluations', [])) > 0:
            has_tabu_info = True
            # Recolectar TODAS las evaluaciones de TODAS las réplicas
            for replica in replicas_results:
                try:
                    replica_mask_data = torch.load(replica['mask_path'], weights_only=False)
                    replica_trace = replica_mask_data['trace']
                    if 'dense_global_best_history' in replica_trace and len(replica_trace['dense_global_best_history']) > 0:
                        # Tomar TODAS las accuracies del historial denso (incluye Tabu)
                        all_accs = [acc * 100 for acc in replica_trace['dense_global_best_history']]
                        tabu_all_accs.extend(all_accs)
                except:
                    pass
    
    # Crear boxplot con o sin Tabu
    if has_tabu_info and len(tabu_all_accs) > 0:
        # Tres categorías: Tabu (todas las evals), Validación Final, Test
        # Similar a GLS que muestra accuracy_history completa
        box_data = [tabu_all_accs, acc_explores, acc_tests]
        if method_name == 'GA_TABU_NSGA2':
            labels = ['GA+Tabu\n(todas evals)', 'Final\n(validación)', 'Test']
        else:  # SS
            labels = ['Tabu Search\n(todas evals)', 'SS Final\n(validación)', 'Test']
        colors = ['lightyellow', 'lightblue', 'lightgreen']
    else:
        # Solo dos categorías: Validación y Test (comportamiento original)
        box_data = [acc_explores, acc_tests]
        labels = ['Validación', 'Test']
        colors = ['lightblue', 'lightgreen']
    
    bp = ax_box.boxplot(box_data, tick_labels=labels,
                        patch_artist=True, widths=0.6,
                        medianprops=dict(color='red', linewidth=2.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
    
    # Colorear cada boxplot
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # NO graficar puntos individuales - solo boxplot limpio
    
    # Líneas de media
    if has_tabu_info and len(tabu_all_accs) > 0:
        tabu_mean = np.mean(tabu_all_accs)
        ax_box.axhline(y=tabu_mean,
                      color='darkorange', linestyle='--', linewidth=2, alpha=0.7,
                      label=f'Media Tabu: {tabu_mean:.2f}%')
    
    ax_box.axhline(y=aggregated_stats['acc_explore_mean'] * 100,
                  color='green', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'Media exploración: {aggregated_stats["acc_explore_mean"]*100:.2f}%')
    ax_box.axhline(y=aggregated_stats['acc_test_mean'] * 100,
                  color='purple', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'Media test: {aggregated_stats["acc_test_mean"]*100:.2f}%')
    
    ax_box.set_ylabel('Accuracy (%)', fontsize=11)
    
    title_suffix = " (con Tabu Search)" if has_tabu_info and len(tabu_all_accs) > 0 else ""
    ax_box.set_title(f'Distribución de Accuracies - {len(replicas_results)} Réplicas{title_suffix}\n'
                    f'{aggregated_stats["architecture"]}, ρ={aggregated_stats["sparsity"]}',
                    fontsize=12, fontweight='bold')
    ax_box.legend(loc='lower right', fontsize=9)
    ax_box.grid(True, axis='y', alpha=0.3)
    
    # Estadísticas detalladas
    if has_tabu_info and len(tabu_all_accs) > 0:
        stats_text = (
            f'Tabu Search ({len(tabu_all_accs)} evals):\n'
            f'  Media:   {np.mean(tabu_all_accs):.2f}%\n'
            f'  Std:     {np.std(tabu_all_accs):.2f}%\n'
            f'  Rango:   [{np.min(tabu_all_accs):.2f}% - {np.max(tabu_all_accs):.2f}%]\n'
            f'\n'
            f'Validación:\n'
            f'  Media:   {aggregated_stats["acc_explore_mean"]*100:.2f}%\n'
            f'  Std:     {aggregated_stats["acc_explore_std"]*100:.2f}%\n'
            f'  Rango:   [{aggregated_stats["acc_explore_min"]*100:.2f}% - {aggregated_stats["acc_explore_max"]*100:.2f}%]\n'
            f'\n'
            f'Test:\n'
            f'  Media:   {aggregated_stats["acc_test_mean"]*100:.2f}%\n'
            f'  Std:     {aggregated_stats["acc_test_std"]*100:.2f}%\n'
            f'  Rango:   [{aggregated_stats["acc_test_min"]*100:.2f}% - {aggregated_stats["acc_test_max"]*100:.2f}%]'
        )
    else:
        stats_text = (
            f'Validación:\n'
            f'  Media:   {aggregated_stats["acc_explore_mean"]*100:.2f}%\n'
            f'  Std:     {aggregated_stats["acc_explore_std"]*100:.2f}%\n'
            f'  Rango:   [{aggregated_stats["acc_explore_min"]*100:.2f}% - {aggregated_stats["acc_explore_max"]*100:.2f}%]\n'
            f'\n'
            f'Test:\n'
            f'  Media:   {aggregated_stats["acc_test_mean"]*100:.2f}%\n'
            f'  Std:     {aggregated_stats["acc_test_std"]*100:.2f}%\n'
            f'  Rango:   [{aggregated_stats["acc_test_min"]*100:.2f}% - {aggregated_stats["acc_test_max"]*100:.2f}%]'
        )
    
    ax_box.text(0.02, 0.98, stats_text, transform=ax_box.transAxes,
               fontsize=8, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    plt.tight_layout()
    
    # Guardar figura combinada
    combined_filename = f'replicas_analysis_{config_signature}.png'
    combined_path = os.path.join(output_dir, combined_filename)
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Grafico guardado: {combined_filename}")


def run_testbed(config_path: str = "configs/experiment.yaml"):
    """
    Ejecuta el testbed completo de experimentos.
    
    Args:
        config_path: Ruta al archivo de configuración
    """
    print(f"\n{'='*80}")
    print("STRONG LOTTERY TICKET HYPOTHESIS - TESTBED")
    print(f"{'='*80}\n")
    
    # Cargar configuración
    config = load_config(config_path)
    print(f"Configuración cargada desde: {config_path}")
    
    # Preparar directorios
    logging_cfg = config['logging']
    plots_enabled = logging_cfg.get('plot_convergence', False)
    save_masks = logging_cfg.get('save_masks', True)

    ensure_dir(os.path.dirname(config['output']['results_csv']))
    ensure_dir(config['output']['logs_dir'])

    if save_masks:
        ensure_dir(config['output']['masks_dir'])

    if plots_enabled:
        ensure_dir(config['output']['plots_dir'])
    
    # Manejar seed_split (puede ser int único o lista para CV)
    seed_splits = config['dataset']['seed_split']
    if isinstance(seed_splits, int):
        seed_splits = [seed_splits]  # Convertir a lista de 1 elemento
    elif not isinstance(seed_splits, list):
        raise ValueError(f"seed_split debe ser int o lista, recibido: {type(seed_splits)}")
    
    print(f"\nValidación Cruzada: {len(seed_splits)} splits")
    print(f"   Seeds: {seed_splits}")
    
    # NOTA: Por ahora ejecutaremos con el PRIMER split
    # TODO: Implementar loop completo sobre todos los splits
    seed_split_to_use = seed_splits[0]
    
    if len(seed_splits) > 1:
        print(f"\n[!] ADVERTENCIA: Detectados {len(seed_splits)} splits")
        print(f"   Por ahora solo se usará el primero: {seed_split_to_use}")
        print(f"   Los demás splits ({seed_splits[1:]}) serán ignorados")
        print(f"   TODO: Implementar loop completo para CV")
    
    # Cargar datos
    print("\n[1/4] Cargando datos...")
    with Timer("Carga de datos"):
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_dataset(
            config['dataset'].get('name', 'wine'),
            seed_split=seed_split_to_use,  # Usar solo el primero por ahora
            train_ratio=config['dataset']['train_ratio'],
            val_ratio=config['dataset']['val_ratio'],
            test_ratio=config['dataset']['test_ratio'],
            path=config['dataset'].get('path'),
            dedup=config['dataset'].get('dedup', False)
        )
    
    data_info = get_data_info(X_train, y_train, X_val, y_val, X_test, y_test)
    print(f"\nInfo del dataset (split seed={seed_split_to_use}):")
    for key, value in data_info.items():
        print(f"  {key}: {value}")
    
    # Generar instancias del testbed
    print("\n[2/4] Generando instancias del testbed...")
    instances = generate_testbed_instances(config)
    total_instances = len(instances)
    
    num_replicas = config.get('num_replicas', 11)
    total_runs = total_instances * num_replicas

    print(f"Total de configuraciones únicas: {total_instances}")
    print(f"  Arquitecturas: {len(config['architectures'])}")
    print(f"  Sparsities: {len(config['sparsities'])}")
    print(f"  Seed models: {len(config['seed_model'])}")
    print(f"  Seed searches: {len(config['seed_search'])}")
    print(f"\nReplicas por configuracion: {num_replicas}")
    print(f"Total de ejecuciones: {total_runs:,}")

    # Preparar archivos de salida
    results_individual_csv = config['output']['results_csv'].replace('.csv', '_individual.csv')
    results_aggregated_csv = config['output']['results_csv'].replace('.csv', '_aggregated.csv')

    print(f"\nArchivos de salida:")
    print(f"   Individuales: {results_individual_csv}")
    print(f"   Agregados:    {results_aggregated_csv}")
    
    # Ejecutar instancias
    print(f"\n[3/4] Ejecutando experimentos...")
    print(f"{'='*80}\n")
    
    # Inicializar loggers detallados
    obj_logger = ObjectiveFunctionLogger()
    strategy_logger = StrategyResultsLogger()
    
    # Variables para estimación de tiempo
    import time as time_module
    start_time_exp = time_module.time()
    
    all_individual_results = []
    all_aggregated_results = []
    
    for i, instance in enumerate(instances, 1):
        # Calcular progreso y tiempo estimado
        progress_pct = (i / total_instances) * 100
        remaining = total_instances - i
        
        # Estimación de tiempo restante
        if i > 1:
            elapsed_time = time_module.time() - start_time_exp
            avg_time_per_instance = elapsed_time / (i - 1)
            estimated_remaining_sec = avg_time_per_instance * remaining
            
            # Convertir a formato legible
            hours = int(estimated_remaining_sec // 3600)
            minutes = int((estimated_remaining_sec % 3600) // 60)
            seconds = int(estimated_remaining_sec % 60)
            
            if hours > 0:
                time_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"
            
            print(f"\n{'='*80}")
            print(f"PROGRESO: Configuracion {i}/{total_instances} ({progress_pct:.1f}%)")
            print(f"Tiempo promedio: {avg_time_per_instance:.1f}s/config ({avg_time_per_instance/num_replicas:.1f}s/replica)")
            print(f"Tiempo estimado restante: {time_str}")
            print(f"{'='*80}")
        else:
            print(f"\n{'='*80}")
            print(f"PROGRESO: Configuracion {i}/{total_instances} ({progress_pct:.1f}%)")
            print(f"{'='*80}")
        
        try:
            with Timer(f"Configuración {i} ({num_replicas} réplicas)"):
                # Ejecutar replicas
                replicas_results, aggregated_stats = run_instance_with_replicas(
                    instance=instance,
                    config=config,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    num_replicas=num_replicas,
                    verbose=config['logging']['verbose'],
                    obj_logger=obj_logger,
                    strategy_logger=strategy_logger
                )
            
            # Guardar resultados individuales inmediatamente
            for run_data in replicas_results:
                save_run_to_csv(run_data, results_individual_csv)
                all_individual_results.append(run_data)
            
            # Guardar resultado agregado
            if aggregated_stats:
                save_run_to_csv(aggregated_stats, results_aggregated_csv)
                all_aggregated_results.append(aggregated_stats)

                if plots_enabled:
                    config_sig = f"arch_{aggregated_stats['architecture'].replace('[', '').replace(']', '').replace(', ', '-')}_sp{aggregated_stats['sparsity']}_sm{aggregated_stats['seed_model']}"
                    plot_replicas_analysis(
                        replicas_results=replicas_results,
                        aggregated_stats=aggregated_stats,
                        output_dir=config['output']['plots_dir'],
                        config_signature=config_sig
                    )

            # Guardar logs incrementalmente cada N configuraciones
            if i % 5 == 0 or i == total_instances:
                logs_dir = config['output']['logs_dir']
                ensure_dir(logs_dir)

                obj_logger_path = os.path.join(logs_dir, 'llamadas_funcion_objetivo.csv')
                obj_logger.save_to_csv(obj_logger_path)

                print(f"\nLogs guardados (checkpoint en config {i}/{total_instances})")
            
        except Exception as e:
            print(f"\nERROR en configuracion {i}:")
            print(f"  {str(e)}")
            traceback.print_exc()
            continue
    
    # Generar reporte final
    print(f"\n[4/4] Generando reporte final...")
    
    # Guardar logs detallados
    logs_dir = config['output']['logs_dir']
    ensure_dir(logs_dir)
    
    obj_logger_path = os.path.join(logs_dir, 'llamadas_funcion_objetivo.csv')
    obj_logger.save_to_csv(obj_logger_path)
    
    print(f"\n{'='*80}")
    print("LOGS DETALLADOS GUARDADOS")
    print(f"{'='*80}")
    print(f"\nLlamadas Funcion Objetivo: {obj_logger_path}")
    print(f"  Total de llamadas: {len(obj_logger.calls):,}")

    print(f"\nResultados Individuales: {results_individual_csv}")
    print(f"  Total de ejecuciones: {len(all_individual_results):,}")
    print(f"  Incluye: replica_id, memoria_mb, solution_repr")

    print(f"\nResultados Agregados: {results_aggregated_csv}")
    print(f"  Total de configuraciones: {len(all_aggregated_results):,}")
    print(f"  Incluye: mean, median, std, min, max para acc_explore y acc_test")

    # Desglose del experimento
    print(f"\nDESGLOSE DEL EXPERIMENTO:")
    print(f"  Arquitecturas: {len(config['architectures'])}")
    print(f"  Niveles de sparsity: {len(config['sparsities'])}")
    print(f"  Seeds modelo: {len(config['seed_model'])}")
    print(f"  Seeds busqueda: {len(config['seed_search'])}")
    print(f"  Replicas por configuracion: {num_replicas}")
    print(f"  Total de configuraciones: {total_instances:,}")
    print(f"  Configuraciones completadas: {len(all_aggregated_results):,}")
    print(f"  Total de ejecuciones: {len(all_individual_results):,}")

    # Calcular tiempo total
    total_time_exp = time_module.time() - start_time_exp
    hours_total = int(total_time_exp // 3600)
    minutes_total = int((total_time_exp % 3600) // 60)
    seconds_total = int(total_time_exp % 60)
    print(f"\nTIEMPO TOTAL DE EJECUCION: ", end="")
    if hours_total > 0:
        print(f"{hours_total}h {minutes_total}m {seconds_total}s")
    elif minutes_total > 0:
        print(f"{minutes_total}m {seconds_total}s")
    else:
        print(f"{seconds_total}s")
    
    if len(all_aggregated_results) > 0:
        df_aggregated = pd.DataFrame(all_aggregated_results)
        
        # Estadísticas generales de resultados agregados
        print(f"\n{'='*80}")
        print("RESUMEN DE RESULTADOS (AGREGADOS POR CONFIGURACIÓN)")
        print(f"{'='*80}")
        print(f"\nTotal de configuraciones exitosas: {len(df_aggregated)}")
        print(f"\nAccuracy exploración (promedio de promedios):")
        print(f"  Media:   {df_aggregated['acc_explore_mean'].mean()*100:.4f}%")
        print(f"  Mediana: {df_aggregated['acc_explore_mean'].median()*100:.4f}%")
        print(f"  Mejor configuración (mean): {df_aggregated['acc_explore_mean'].max()*100:.4f}%")
        print(f"  Peor configuración (mean):  {df_aggregated['acc_explore_mean'].min()*100:.4f}%")
        
        print(f"\nAccuracy test (promedio de promedios):")
        print(f"  Media:   {df_aggregated['acc_test_mean'].mean()*100:.4f}%")
        print(f"  Mediana: {df_aggregated['acc_test_mean'].median()*100:.4f}%")
        print(f"  Mejor configuración (mean): {df_aggregated['acc_test_mean'].max()*100:.4f}%")
        print(f"  Peor configuración (mean):  {df_aggregated['acc_test_mean'].min()*100:.4f}%")
        
        # Top 5 configuraciones por accuracy validation media
        print(f"\n{'='*80}")
        print("TOP 5 CONFIGURACIONES (por acc_explore_mean):")
        print(f"{'='*80}")
        top5 = df_aggregated.nlargest(5, 'acc_explore_mean')[
            ['architecture', 'sparsity', 'seed_model', 'acc_explore_mean', 'acc_explore_std', 
             'acc_test_mean', 'acc_test_std', 'num_replicas']
        ].copy()
        
        # Formatear porcentajes
        for col in ['acc_explore_mean', 'acc_explore_std', 'acc_test_mean', 'acc_test_std']:
            top5[col] = top5[col] * 100
        
        print(top5.to_string(index=False))
        
        # Generar visualizaciones con datos agregados
        generate_report(df_aggregated, config['output']['plots_dir'])
        
        print(f"\n{'='*80}")
        print(f"Resultados individuales: {results_individual_csv}")
        print(f"Resultados agregados:    {results_aggregated_csv}")
        print(f"Mascaras guardadas en:   {config['output']['masks_dir']}")
        if plots_enabled:
            print(f"Plots guardados en:      {config['output']['plots_dir']}")
        print(f"{'='*80}\n")
    
    else:
        print("\nNo se completaron configuraciones exitosas")


if __name__ == "__main__":
    # Permitir especificar config por línea de comandos
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/experiment.yaml"
    
    run_testbed(config_path)
