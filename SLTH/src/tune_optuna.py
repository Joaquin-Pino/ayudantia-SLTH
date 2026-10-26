"""
Optimización de hiperparámetros con Optuna para GLS, Scatter Search y GA+TABU+NSGA2.

Uso:
    python src/tune_optuna.py gls              [--config configs/optuna_gls.yaml] [--output-dir results/optuna_gls]
    python src/tune_optuna.py ss               [--config configs/optuna_ss.yaml]  [--output-dir results/optuna_ss]
    python src/tune_optuna.py ga_tabu_nsga2    [--config configs/optuna_ga_tabu_nsga2.yaml] [--output-dir results/optuna_ga_tabu_nsga2]

Para otro dataset (p. ej. BCW), el config incluye una seccion 'dataset' con el mismo split del experimento:
    python src/tune_optuna.py gls --config configs/optuna_gls_bcw.yaml --output-dir results/bcw/optuna_gls
"""

import argparse
import json
import os

import numpy as np
import optuna
import yaml
from optuna.trial import Trial
from typing import List

from data_loader import load_dataset, load_wine_data
from gls import run_gls_experiment
from ga_tabu_nsga2 import run_ga_tabu_nsga2_experiment
from mask import create_mask
from model import build_mlp
from scatter_search import run_ss_experiment
from seeds import set_seed


def _load_data(dataset: dict | None, seed_base: int):
    """
    Carga los datos de la seccion 'dataset' del config, con el mismo split que el
    experimento: Optuna puntua con el conjunto de busqueda (val) y no toca test.
    Sin esa seccion se mantiene el comportamiento original (Wine, seed_split=seed_base).
    """
    if not dataset:
        return load_wine_data(seed_split=seed_base)
    return load_dataset(
        dataset.get('name', 'wine'),
        seed_split=dataset.get('seed_split', seed_base),
        train_ratio=dataset['train_ratio'],
        val_ratio=dataset['val_ratio'],
        test_ratio=dataset['test_ratio'],
        path=dataset.get('path'),
        dedup=dataset.get('dedup', False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GLS
# ─────────────────────────────────────────────────────────────────────────────

class GLSOptunaOptimizer:
    def __init__(self, architectures: List[List[int]], sparsities: List[float],
                 n_trials: int = 20, seed_base: int = 42,
                 study_name: str = "gls_optimization",
                 dataset: dict | None = None):
        self.architectures = architectures
        self.sparsities = sparsities
        self.n_trials = n_trials
        self.seed_base = seed_base
        self.study_name = study_name

        print("Cargando datos...")
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.scaler = \
            _load_data(dataset, seed_base)
        print(" Datos cargados")

    def objective(self, trial: Trial) -> float:
        alpha = trial.suggest_float('alpha', 0.5, 0.99, step=0.05)
        lambda_param = trial.suggest_float('lambda', 0.01, 0.5, step=0.01)
        max_iter = trial.suggest_int('max_iter', 500, 2000, step=100)
        max_no_improve = trial.suggest_int('max_no_improve', 50, 200, step=25)
        ls_max_neighbors = trial.suggest_int('ls_max_neighbors', 50, 200, step=25)
        ls_max_iters = trial.suggest_int('ls_max_iters', 3, 15, step=1)

        hp_gls = {
            'alpha': alpha, 'lambda': lambda_param, 'max_iter': max_iter,
            'max_evals': 10000, 'max_no_improve': max_no_improve,
            'ls_max_neighbors': ls_max_neighbors, 'ls_max_iters': ls_max_iters,
        }

        seed_trial = self.seed_base + trial.number
        np.random.seed(seed_trial)
        architecture = self.architectures[np.random.randint(len(self.architectures))]
        sparsity = self.sparsities[np.random.randint(len(self.sparsities))]

        model = build_mlp(architecture, seed_model=seed_trial)
        shapes = model.get_weight_shapes()
        mask_init = create_mask(shapes, sparsity=sparsity, per_layer=True, seed=seed_trial)

        try:
            results = run_gls_experiment(
                model=model, mask_init=mask_init,
                X_val=self.X_val, y_val=self.y_val,
                X_test=self.X_test, y_test=self.y_test,
                hp_gls=hp_gls, seed_search=seed_trial, verbose=False, obj_logger=None
            )
            acc_explore = results['acc_explore']
            trial.set_user_attr('acc_test', results['acc_test'])
            trial.set_user_attr('evaluations', results['trace']['final_stats']['total_evaluations'])
            trial.set_user_attr('time_seconds', results['trace']['final_stats']['total_time'])
            trial.set_user_attr('active_connections', results['trace']['final_stats']['active_connections'])
            return acc_explore
        except Exception as e:
            print(f"Trial {trial.number} falló: {e}")
            return 0.0

    def optimize(self) -> optuna.Study:
        print(f"\n{'='*80}")
        print("OPTIMIZACIÓN DE HIPERPARÁMETROS GLS CON OPTUNA")
        print(f"Arquitecturas: {self.architectures}  Sparsities: {self.sparsities}")
        print(f"Trials: {self.n_trials}\n{'='*80}\n")
        study = optuna.create_study(
            study_name=self.study_name, direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed_base)
        )
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        return study

    def print_results(self, study: optuna.Study):
        print(f"\n{'='*80}\nRESULTADOS DE OPTIMIZACIÓN GLS\n{'='*80}")
        print(f"Trials completados: {len(study.trials)}")
        print("\nMejores hiperparámetros:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        bt = study.best_trial
        print(f"\nAcc validación: {study.best_value*100:.4f}%")
        print(f"Acc test: {bt.user_attrs['acc_test']*100:.4f}%")
        print(f"Evaluaciones: {bt.user_attrs['evaluations']}")
        print(f"Tiempo: {bt.user_attrs['time_seconds']:.2f}s\n{'='*80}\n")

    def save_best_params(self, study: optuna.Study, output_path: str):
        bp = study.best_params
        bt = study.best_trial
        cfg = {
            'GLS': {
                'alpha': bp['alpha'], 'lambda': bp['lambda'],
                'max_iter': bp['max_iter'], 'max_evals': 10000,
                'max_no_improve': bp['max_no_improve'],
                'ls_max_neighbors': bp['ls_max_neighbors'],
                'ls_max_iters': bp['ls_max_iters'],
            },
            'optimization_results': {
                'best_acc_explore': float(study.best_value),
                'best_acc_test': float(bt.user_attrs['acc_test']),
                'n_trials': len(study.trials),
                'architectures_used': self.architectures,
                'sparsities_used': self.sparsities,
            }
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f" Parámetros guardados en: {output_path}")

    def save_study_results(self, study: optuna.Study, output_dir: str):
        _save_study_artifacts(study, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Scatter Search
# ─────────────────────────────────────────────────────────────────────────────

class SSOptunaOptimizer:
    def __init__(self, architectures: List[List[int]], sparsities: List[float],
                 n_trials: int = 20, seed_base: int = 42,
                 study_name: str = "ss_optimization",
                 dataset: dict | None = None):
        self.architectures = architectures
        self.sparsities = sparsities
        self.n_trials = n_trials
        self.seed_base = seed_base
        self.study_name = study_name

        print("Cargando datos...")
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.scaler = \
            _load_data(dataset, seed_base)
        print(" Datos cargados")

    def objective(self, trial: Trial) -> float:
        refset_size = trial.suggest_int('refset_size', 4, 20, step=2)
        diversification_size = trial.suggest_int('diversification_size', 50, 200, step=25)
        ts_tenure = trial.suggest_int('ts_tenure', 5, 20, step=1)
        ts_max_iter = trial.suggest_int('ts_max_iter', 10, 50, step=5)
        ts_max_no_improve = trial.suggest_int('ts_max_no_improve', 5, 25, step=5)
        max_no_improve = trial.suggest_int('max_no_improve', 5, 50, step=5)
        max_iter = trial.suggest_int('max_iter', 10, 100, step=10)

        hp_ss = {
            'refset_size': refset_size, 'refset_quality_ratio': 0.5,
            'diversification_size': diversification_size,
            'ts_tenure': ts_tenure, 'ts_max_iter': ts_max_iter,
            'ts_max_no_improve': ts_max_no_improve,
            'max_iter': max_iter, 'max_evals': 10000,
            'max_no_improve': max_no_improve, 'min_diversity': None,
        }

        seed_trial = self.seed_base + trial.number
        np.random.seed(seed_trial)
        architecture = self.architectures[np.random.randint(len(self.architectures))]
        sparsity = self.sparsities[np.random.randint(len(self.sparsities))]

        model = build_mlp(architecture, seed_model=seed_trial)
        shapes = model.get_weight_shapes()
        mask_init = create_mask(shapes, sparsity=sparsity, per_layer=True, seed=seed_trial)

        try:
            results = run_ss_experiment(
                model=model, mask_init=mask_init,
                X_val=self.X_val, y_val=self.y_val,
                X_test=self.X_test, y_test=self.y_test,
                hp_ss=hp_ss, seed_search=seed_trial, verbose=False, obj_logger=None
            )
            acc_explore = results['acc_explore']
            trial.set_user_attr('acc_test', results['acc_test'])
            trial.set_user_attr('evaluations', results['trace']['final_stats']['total_evaluations'])
            trial.set_user_attr('time_seconds', results['trace']['final_stats']['total_time'])
            trial.set_user_attr('active_connections', results['trace']['final_stats']['active_connections'])
            trial.set_user_attr('iterations', results['trace']['final_stats']['iterations'])
            return acc_explore
        except Exception as e:
            print(f"Trial {trial.number} falló: {e}")
            return 0.0

    def optimize(self) -> optuna.Study:
        print(f"\n{'='*80}")
        print("OPTIMIZACIÓN DE HIPERPARÁMETROS SCATTER SEARCH CON OPTUNA")
        print(f"Arquitecturas: {self.architectures}  Sparsities: {self.sparsities}")
        print(f"Trials: {self.n_trials}\n{'='*80}\n")
        study = optuna.create_study(
            study_name=self.study_name, direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed_base)
        )
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        return study

    def print_results(self, study: optuna.Study):
        print(f"\n{'='*80}\nRESULTADOS DE OPTIMIZACIÓN SS\n{'='*80}")
        print(f"Trials completados: {len(study.trials)}")
        print("\nMejores hiperparámetros:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        bt = study.best_trial
        print(f"\nAcc validación: {study.best_value*100:.4f}%")
        print(f"Acc test: {bt.user_attrs['acc_test']*100:.4f}%")
        print(f"Evaluaciones: {bt.user_attrs['evaluations']}")
        print(f"Iteraciones: {bt.user_attrs['iterations']}")
        print(f"Tiempo: {bt.user_attrs['time_seconds']:.2f}s\n{'='*80}\n")

    def save_best_params(self, study: optuna.Study, output_path: str):
        bp = study.best_params
        bt = study.best_trial
        cfg = {
            'SS': {
                'refset_size': bp['refset_size'], 'refset_quality_ratio': 0.5,
                'diversification_size': bp['diversification_size'],
                'ts_tenure': bp['ts_tenure'], 'ts_max_iter': bp['ts_max_iter'],
                'ts_max_no_improve': bp['ts_max_no_improve'],
                'max_iter': bp['max_iter'], 'max_evals': 10000,
                'max_no_improve': bp['max_no_improve'], 'min_diversity': None,
            },
            'optimization_results': {
                'best_acc_explore': float(study.best_value),
                'best_acc_test': float(bt.user_attrs['acc_test']),
                'n_trials': len(study.trials),
                'architectures_used': self.architectures,
                'sparsities_used': self.sparsities,
            }
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f" Parámetros guardados en: {output_path}")

    def save_study_results(self, study: optuna.Study, output_dir: str):
        _save_study_artifacts(study, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# GA + TABU + NSGA-II
# ─────────────────────────────────────────────────────────────────────────────

class GATabuNSGA2OptunaOptimizer:
    # sparsities no se usa: GA+NSGA2 explora el rango 0.1–0.9 como objetivo, no como HP.
    def __init__(self, architectures: List[List[int]],
                 n_trials: int = 20, seed_base: int = 42,
                 study_name: str = "ga_tabu_nsga2_optimization",
                 dataset: dict | None = None):
        self.architectures = architectures
        self.n_trials = n_trials
        self.seed_base = seed_base
        self.study_name = study_name

        print("Cargando datos...")
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.scaler = \
            _load_data(dataset, seed_base)
        print(" Datos cargados")

    def _calculate_hypervolume(self, f1_values: List[float], f2_values: List[float],
                                ref_point: tuple = (1.0, 1.0)) -> float:
        if len(f1_values) == 0:
            return 0.0
        f1_ref, f2_ref = ref_point
        valid_points = [(f1, f2) for f1, f2 in zip(f1_values, f2_values) if f1 < f1_ref and f2 < f2_ref]
        if not valid_points:
            return 0.0
        points = sorted(valid_points, key=lambda x: x[0])
        # Barrido en f1: cada franja [f1, next_f1) queda dominada desde el menor f2
        # visto hasta ahí hasta f2_ref (los puntos dominados no suman área)
        hv = 0.0
        min_f2 = f2_ref
        for i, (f1, f2) in enumerate(points):
            next_f1 = points[i + 1][0] if i < len(points) - 1 else f1_ref
            min_f2 = min(min_f2, f2)
            hv += (next_f1 - f1) * (f2_ref - min_f2)
        return hv

    def objective(self, trial: Trial) -> float:
        pop_size = trial.suggest_int('pop_size', 20, 100, step=10)
        max_generations = trial.suggest_int('max_generations', 50, 200, step=25)
        mutation_rate = round(trial.suggest_float('mutation_rate', 0.001, 0.05), 5)
        crossover_rate = round(trial.suggest_float('crossover_rate', 0.7, 0.95), 5)
        tournament_size = trial.suggest_int('tournament_size', 2, 5, step=1)
        tabu_tenure = trial.suggest_int('tabu_tenure', 3, 15, step=1)
        tabu_max_iter = trial.suggest_int('tabu_max_iter', 10, 50, step=5)
        tabu_max_no_improve = trial.suggest_int('tabu_max_no_improve', 5, 25, step=5)
        apply_tabu_every = trial.suggest_int('apply_tabu_every', 5, 20, step=5)

        seed_trial = self.seed_base + trial.number
        np.random.seed(seed_trial)
        architecture = self.architectures[np.random.randint(len(self.architectures))]

        model = build_mlp(architecture, seed_model=seed_trial)
        shapes = model.get_weight_shapes()
        mask_init = create_mask(shapes, sparsity=0.5, per_layer=True, seed=seed_trial)

        try:
            results = run_ga_tabu_nsga2_experiment(
                model=model, mask_init=mask_init,
                X_val=self.X_val, y_val=self.y_val,
                X_test=self.X_test, y_test=self.y_test,
                pop_size=pop_size, max_generations=max_generations,
                mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                tournament_size=tournament_size,
                min_sparsity=0.1, max_sparsity=0.9,
                tabu_tenure=tabu_tenure, tabu_max_iter=tabu_max_iter,
                tabu_max_no_improve=tabu_max_no_improve,
                apply_tabu_every=apply_tabu_every,
                max_evals=10000, plot_convergence=False,
                seed=seed_trial, verbose=False
            )

            hypervolume = 0.0
            best_acc_explore = 0.0
            pareto_accs: List[float] = []

            if 'trace' in results and 'pareto_front_history' in results['trace']:
                pareto_history = results['trace']['pareto_front_history']
                if pareto_history:
                    _, pareto_f1s, pareto_f2s, pareto_accs, _ = pareto_history[-1]
                    hypervolume = self._calculate_hypervolume(pareto_f1s, pareto_f2s)
                    best_acc_explore = max(pareto_accs) if pareto_accs else 0.0
                    print(f"   Trial {trial.number}: HV={hypervolume:.5f}, "
                          f"Pareto size={len(pareto_f1s)}, Best Acc={best_acc_explore:.4f}")

            trial.set_user_attr('hypervolume', round(hypervolume, 5))
            trial.set_user_attr('best_acc_explore', round(best_acc_explore, 5))
            trial.set_user_attr('best_acc_test', round(results.get('acc_test', 0.0), 5))
            trial.set_user_attr('best_sparsity', round(results.get('sparsity', 0.0), 5))
            trial.set_user_attr('total_evaluations', results.get('total_evaluations', 0))
            trial.set_user_attr('total_time', round(results.get('time', 0.0), 2))
            trial.set_user_attr('pareto_front_size', len(pareto_accs))
            return hypervolume

        except Exception as e:
            print(f"Trial {trial.number} falló: {e}")
            return 0.0

    def optimize(self) -> optuna.Study:
        print(f"\n{'='*80}")
        print("OPTIMIZACIÓN DE HIPERPARÁMETROS GA+TABU+NSGA2 CON OPTUNA")
        print(f"Arquitecturas: {len(self.architectures)}  Métrica: Hipervolumen")
        print(f"Trials: {self.n_trials}\n{'='*80}\n")
        study = optuna.create_study(
            study_name=self.study_name, direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed_base)
        )
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        return study

    def print_results(self, study: optuna.Study):
        print(f"\n{'='*80}\nRESULTADOS DE OPTIMIZACIÓN GA+TABU+NSGA2\n{'='*80}")
        print(f"Trials completados: {len(study.trials)}")
        print("\nMejores hiperparámetros:")
        for k, v in study.best_params.items():
            print(f"  {k}: {round(v, 5) if isinstance(v, float) else v}")
        bt = study.best_trial
        print(f"\nHipervolumen: {study.best_value:.5f}")
        print(f"Mejor acc exploración: {bt.user_attrs['best_acc_explore']*100:.4f}%")
        print(f"Mejor acc test: {bt.user_attrs['best_acc_test']*100:.4f}%")
        print(f"Tamaño frente Pareto: {bt.user_attrs['pareto_front_size']}")
        print(f"Tiempo: {bt.user_attrs['total_time']:.2f}s\n{'='*80}\n")

    def save_best_params(self, study: optuna.Study, output_path: str):
        bp = study.best_params
        bt = study.best_trial
        cfg = {
            'GA_TABU_NSGA2': {
                'pop_size': int(bp['pop_size']),
                'max_generations': int(bp['max_generations']),
                'mutation_rate': round(bp['mutation_rate'], 5),
                'crossover_rate': round(bp['crossover_rate'], 5),
                'tournament_size': int(bp['tournament_size']),
                'min_sparsity': 0.1,
                'max_sparsity': 0.9,
                'tabu_tenure': int(bp['tabu_tenure']),
                'tabu_max_iter': int(bp['tabu_max_iter']),
                'tabu_max_no_improve': int(bp['tabu_max_no_improve']),
                'apply_tabu_every': int(bp['apply_tabu_every']),
                'max_evals': 10000,
            },
            'optimization_results': {
                'hypervolume': round(study.best_value, 5),
                'best_acc_explore': round(bt.user_attrs['best_acc_explore'], 5),
                'best_acc_test': round(bt.user_attrs['best_acc_test'], 5),
                'pareto_front_size': int(bt.user_attrs['pareto_front_size']),
                'n_trials': len(study.trials),
                'architectures_used': self.architectures,
            }
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f" Parámetros guardados en: {output_path}")

    def save_study_results(self, study: optuna.Study, output_dir: str):
        _save_study_artifacts(study, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad compartida — guardar artefactos del estudio
# ─────────────────────────────────────────────────────────────────────────────

def _save_study_artifacts(study: optuna.Study, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'optuna_trials.csv')
    study.trials_dataframe().to_csv(csv_path, index=False)
    print(f" Trials guardados en: {csv_path}")

    try:
        importance = optuna.importance.get_param_importances(study)
        importance_path = os.path.join(output_dir, 'param_importance.json')
        with open(importance_path, 'w') as f:
            json.dump(importance, f, indent=2)
        print(f" Importancia de parámetros en: {importance_path}")
    except Exception as e:
        print(f"  No se pudo calcular importancia: {e}")

    try:
        import plotly  # noqa: F401
        for name, fn in [
            ('optimization_history', optuna.visualization.plot_optimization_history),
            ('parallel_coordinate', optuna.visualization.plot_parallel_coordinate),
            ('param_importances', optuna.visualization.plot_param_importances),
        ]:
            try:
                fn(study).write_html(os.path.join(output_dir, f'{name}.html'))
                print(f" {name}.html guardado")
            except Exception as e:
                print(f"  No se pudo crear {name}: {e}")
    except ImportError:
        print("  plotly no disponible, omitiendo visualizaciones HTML")


# ─────────────────────────────────────────────────────────────────────────────
# Funciones de entrada por método
# ─────────────────────────────────────────────────────────────────────────────

def run_gls(config_path: str, output_dir: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    opt = config['optimization']
    optimizer = GLSOptunaOptimizer(
        architectures=opt['architectures'], sparsities=opt['sparsities'],
        n_trials=opt['n_trials'], seed_base=opt['seed_base'],
        study_name=opt.get('study_name', 'gls_optimization'),
        dataset=config.get('dataset'),
    )
    study = optimizer.optimize()
    optimizer.print_results(study)
    os.makedirs(output_dir, exist_ok=True)
    optimizer.save_best_params(study, os.path.join(output_dir, 'best_gls_params.yaml'))
    optimizer.save_study_results(study, output_dir)
    print(f"\n Optimización GLS completada. Resultados en: {output_dir}\n")


def run_ss(config_path: str, output_dir: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    opt = config['optimization']
    optimizer = SSOptunaOptimizer(
        architectures=opt['architectures'], sparsities=opt['sparsities'],
        n_trials=opt['n_trials'], seed_base=opt['seed_base'],
        study_name=opt.get('study_name', 'ss_optimization'),
        dataset=config.get('dataset'),
    )
    study = optimizer.optimize()
    optimizer.print_results(study)
    os.makedirs(output_dir, exist_ok=True)
    optimizer.save_best_params(study, os.path.join(output_dir, 'best_ss_params.yaml'))
    optimizer.save_study_results(study, output_dir)
    print(f"\n Optimización SS completada. Resultados en: {output_dir}\n")


def run_ga_tabu_nsga2(config_path: str, output_dir: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    opt = config['optimization']
    optimizer = GATabuNSGA2OptunaOptimizer(
        architectures=opt['architectures'],
        n_trials=opt['n_trials'], seed_base=opt['seed_base'],
        study_name=opt.get('study_name', 'ga_tabu_nsga2_optimization'),
        dataset=config.get('dataset'),
    )
    study = optimizer.optimize()
    optimizer.print_results(study)
    os.makedirs(output_dir, exist_ok=True)
    optimizer.save_best_params(study, os.path.join(output_dir, 'best_ga_tabu_nsga2_params.yaml'))
    optimizer.save_study_results(study, output_dir)
    print(f"\n Optimización GA+TABU+NSGA2 completada. Resultados en: {output_dir}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI unificado
# ─────────────────────────────────────────────────────────────────────────────

_METHODS = {
    'gls':           (run_gls,           'configs/optuna_gls.yaml',           'results/optuna_gls'),
    'ss':            (run_ss,            'configs/optuna_ss.yaml',            'results/optuna_ss'),
    'ga_tabu_nsga2': (run_ga_tabu_nsga2, 'configs/optuna_ga_tabu_nsga2.yaml', 'results/optuna_ga_tabu_nsga2'),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimización de hiperparámetros con Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('method', choices=list(_METHODS), help="Método a optimizar")
    parser.add_argument('--config', type=str, default=None, help="Ruta al config YAML")
    parser.add_argument('--output-dir', type=str, default=None, help="Directorio de salida")
    parser.add_argument('--n-trials', type=int, default=None,
                        help="Sobrescribir n_trials del config")
    args = parser.parse_args()

    fn, default_config, default_output = _METHODS[args.method]
    config_path = args.config or default_config
    output_dir = args.output_dir or default_output

    # Permitir sobrescribir n_trials desde la CLI
    if args.n_trials is not None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg['optimization']['n_trials'] = args.n_trials
        import tempfile, atexit
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(cfg, tmp, default_flow_style=False, sort_keys=False)
        tmp.close()
        atexit.register(os.unlink, tmp.name)
        config_path = tmp.name

    fn(config_path, output_dir)
