"""
Profiler de los métodos de búsqueda (GLS / SS / GA_TABU_NSGA2).

Corre UNA réplica de UNA instancia con cProfile y muestra dónde se va el tiempo,
para comparar métodos y arquitecturas. No modifica ningún archivo del proyecto
ni escribe resultados: usa run_single_experiment sin loggers.

Uso (desde SLTH/):
    ../.venv/bin/python -u profile_methods.py configs/50k/bcw/2-experiment_ss_bcw.yaml \
        --arch-idx 3 --max-evals 1000

Opciones:
    --arch-idx    índice en `architectures` del YAML (default 0)
    --sparsity    sparsity a usar (default: la primera del YAML)
    --max-evals   presupuesto de evaluaciones para la prueba (default 1000)
    --top         cuántas funciones mostrar (default 15)
    --save        guarda el perfil crudo (.prof) en esa ruta, para abrirlo con snakeviz/pstats
"""

import argparse
import cProfile
import copy
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from run_experiment import load_config, generate_testbed_instances, run_single_experiment
from data_loader import load_dataset

# Funciones que se sospecha dominan el tiempo; se reportan aparte
FUNCIONES_SOSPECHOSAS = [
    '_repair_layer_connectivity',
    'repair_mask_connectivity',
    'ensure_input_output_path',
    'swap_1to1',
    'mask_to_move',
    'score',
    'log_call',
    'tabu_search',
    'tabu_search_improvement',
]


def reducir_config(config, arch_idx, sparsity, max_evals):
    """Deja una sola arquitectura/sparsity y fija max_evals del método."""
    cfg = copy.deepcopy(config)
    cfg['architectures'] = [cfg['architectures'][arch_idx]]
    cfg['sparsities'] = [sparsity if sparsity is not None else cfg['sparsities'][0]]
    cfg['seed_model'] = cfg['seed_model'][:1]
    cfg['seed_search'] = cfg['seed_search'][:1]
    cfg[cfg['method']]['max_evals'] = max_evals
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('--arch-idx', type=int, default=0)
    parser.add_argument('--sparsity', type=float, default=None)
    parser.add_argument('--max-evals', type=int, default=1000)
    parser.add_argument('--top', type=int, default=15)
    parser.add_argument('--save', default=None)
    args = parser.parse_args()

    config = reducir_config(load_config(args.config), args.arch_idx, args.sparsity, args.max_evals)
    instance = generate_testbed_instances(config)[0]

    ds = config['dataset']
    _, X_val, X_test, _, y_val, y_test, _ = load_dataset(
        ds.get('name', 'wine'),
        seed_split=ds['seed_split'][0] if isinstance(ds['seed_split'], list) else ds['seed_split'],
        train_ratio=ds['train_ratio'], val_ratio=ds['val_ratio'], test_ratio=ds['test_ratio'],
        path=ds.get('path'), dedup=ds.get('dedup', False),
    )

    print(f"Método: {instance['method']} | arquitectura: {instance['architecture']} | "
          f"sparsity: {instance['sparsity']} | max_evals: {args.max_evals}")

    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    result = run_single_experiment(instance, config, X_val, y_val, X_test, y_test, verbose=False)
    prof.disable()
    total = time.perf_counter() - t0

    evals = result.get('evaluations')
    print(f"\nTiempo total: {total:.2f}s | evaluaciones reportadas: {evals} | "
          f"acc_val={result['acc_explore']:.4f}")
    if evals:
        print(f"Costo por evaluación: {total / evals * 1000:.2f} ms")
        print(f"Extrapolado a 50.000 evals: {total / evals * 50000 / 3600:.2f} h por réplica")

    if args.save:
        prof.dump_stats(args.save)

    stats = pstats.Stats(prof)
    stats.sort_stats('cumulative')

    print(f"\n=== Top {args.top} por tiempo acumulado ===")
    stats.print_stats(args.top)
    print(f"=== Top {args.top} por tiempo propio ===")
    stats.sort_stats('tottime').print_stats(args.top)

    print("=== Funciones sospechosas (% del tiempo total, acumulado) ===")
    filas = []
    for (archivo, linea, nombre), (cc, nc, tt, ct, callers) in stats.stats.items():
        if nombre in FUNCIONES_SOSPECHOSAS:
            filas.append((ct, nombre, os.path.basename(archivo), nc))
    for ct, nombre, archivo, nc in sorted(filas, reverse=True):
        print(f"  {nombre:30s} {archivo:24s} llamadas={nc:8d}  {ct:8.2f}s  {ct / total * 100:5.1f}%")


if __name__ == "__main__":
    main()
