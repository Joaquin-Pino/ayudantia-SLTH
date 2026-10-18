"""
Utilidades: logging, timers, plots de convergencia.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from typing import Dict, List
from pathlib import Path


def ensure_dir(directory: str):
    """Crea directorio si no existe."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_trace_plot(trace: Dict, acc_test: float, output_path: str, title: str = "Convergencia"):
    """
    Guarda gráfico de convergencia de la búsqueda.
    
    Args:
        trace: Dict con historiales de SA/GLS/Tabu
        output_path: Ruta donde guardar el plot
        title: Título del gráfico
    """
    ensure_dir(os.path.dirname(output_path))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy vs Evaluaciones
    ax1 = axes[0, 0]
    evals = trace['evaluations']
    
    # SA tiene 'current_history' y 'best_history'
    # GLS tiene 'accuracy_history' y 'best_accuracy_history'
    if 'current_history' in trace:
        # SA
        ax1.plot(evals, [acc*100 for acc in trace['current_history']], 
                 label='Current', alpha=1, linewidth=1)
        ax1.plot(evals, [acc*100 for acc in trace['best_history']], 
                 label='Best', linewidth=0.5)
    elif 'accuracy_history' in trace:
        # GLS
        ax1.plot(evals, [acc*100 for acc in trace['accuracy_history']], 
                 label='Local Optima', alpha=1, linewidth=1, marker='o', markersize=3)
        ax1.plot(evals, [acc*100 for acc in trace['best_accuracy_history']], 
                 label='Best', linewidth=2)
    
    ax1.set_xlabel('Evaluaciones')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Convergencia de Accuracy (val)')
    ax1.grid(True, alpha=0.3)
    ax1.scatter([evals[-1]], [acc_test * 100], s=70, color='crimson', marker='o', zorder=5, label='Accuracy (test)')
    ax1.annotate(f'{acc_test*100:.2f}%', xy=(evals[-1], acc_test*100),
        xytext=(5, 5), textcoords='offset points', fontsize=8, color='crimson')
    ax1.legend()


    
    # 2. Temperatura (si aplica)
    if 'temperatures' in trace and len(trace['temperatures']) > 0:
        ax2 = axes[0, 1]
        ax2.plot(trace['temperatures'], linewidth=2)
        ax2.set_xlabel('Nivel de temperatura')
        ax2.set_ylabel('Temperatura')
        ax2.set_title('Enfriamiento de temperatura')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    
    # 3. Tasa de aceptación
    if 'acceptance_rates' in trace and len(trace['acceptance_rates']) > 0:
        ax3 = axes[1, 0]
        ax3.plot([r*100 for r in trace['acceptance_rates']], linewidth=2)
        ax3.set_xlabel('Nivel de temperatura')
        ax3.set_ylabel('Tasa de aceptación (%)')
        ax3.set_title('Tasa de aceptación por nivel')
        ax3.grid(True, alpha=0.3)
    
    # 4. Mejora acumulada
    ax4 = axes[1, 1]
    
    # Determinar qué historia usar (SA vs GLS)
    if 'best_history' in trace:
        best_history = np.array(trace['best_history']) * 100
    elif 'best_accuracy_history' in trace:
        best_history = np.array(trace['best_accuracy_history']) * 100
    else:
        # Si no hay ninguna, ocultar este subplot
        ax4.text(0.5, 0.5, 'No disponible', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_xlabel('Evaluaciones')
        ax4.set_ylabel('Mejora acumulada (%)')
        ax4.set_title('Mejora respecto a inicial')
        best_history = None
    
    if best_history is not None:
        initial_acc = best_history[0]
        improvement = best_history - initial_acc
        ax4.plot(evals, improvement, linewidth=2, color='green')
        ax4.set_xlabel('Evaluaciones')
        ax4.set_ylabel('Mejora acumulada (%)')
        ax4.set_title('Mejora respecto a inicial')
        ax4.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot guardado en: {output_path}")


def save_comparison_plot(traces: Dict[str, Dict], output_path: str, 
                        title: str = "Comparación de métodos"):
    """
    Compara múltiples trazas (distintos métodos o configuraciones).
    
    Args:
        traces: Dict {nombre: trace}
        output_path: Ruta de salida
        title: Título
    """
    ensure_dir(os.path.dirname(output_path))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, trace in traces.items():
        evals = trace['evaluations']
        
        # Determinar qué historia usar (SA vs GLS)
        if 'best_history' in trace:
            best_history = [acc*100 for acc in trace['best_history']]
        elif 'best_accuracy_history' in trace:
            best_history = [acc*100 for acc in trace['best_accuracy_history']]
        else:
            continue  # Saltar si no tiene historial
        
        ax.plot(evals, best_history, label=name, linewidth=2)
    
    ax.set_xlabel('Evaluaciones')
    ax.set_ylabel('Mejor Accuracy (%)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot de comparación guardado en: {output_path}")


def save_run_to_csv(run_data: Dict, csv_path: str):
    """
    Guarda resultados de una corrida en CSV.
    
    Args:
        run_data: Dict con datos de la corrida
        csv_path: Ruta del CSV
    """
    ensure_dir(os.path.dirname(csv_path))
    
    # Convertir a DataFrame
    df_row = pd.DataFrame([run_data])
    
    # Append o crear
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df = df_row
    
    df.to_csv(csv_path, index=False)


def load_runs_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Carga resultados desde CSV.
    
    Args:
        csv_path: Ruta del CSV
    
    Returns:
        DataFrame con corridas
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    
    return pd.read_csv(csv_path)


def summarize_runs(df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """
    Resume corridas agrupando por variables.
    
    Args:
        df: DataFrame con corridas
        group_by: Lista de columnas para agrupar
    
    Returns:
        DataFrame con estadísticas
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary = df.groupby(group_by)[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'])
    
    return summary


def save_json(data: Dict, filepath: str):
    """Guarda diccionario en JSON."""
    ensure_dir(os.path.dirname(filepath))
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict:
    """Carga diccionario desde JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_hyperparameter_tuning(df: pd.DataFrame, hp_name: str, metric: str,
                               output_path: str):
    """
    Visualiza impacto de hiperparámetro en métrica.
    
    Args:
        df: DataFrame con corridas
        hp_name: Nombre del hiperparámetro
        metric: Métrica a graficar (e.g., 'acc_explore')
        output_path: Ruta de salida
    """
    ensure_dir(os.path.dirname(output_path))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Agrupar por hiperparámetro
    grouped = df.groupby(hp_name)[metric].agg(['mean', 'std'])
    
    x = grouped.index
    y_mean = grouped['mean'] * 100
    y_std = grouped['std'] * 100
    
    ax.plot(x, y_mean, marker='o', linewidth=2, markersize=8)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3)
    
    ax.set_xlabel(hp_name)
    ax.set_ylabel(f'{metric} (%)')
    ax.set_title(f'Impacto de {hp_name} en {metric}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot de tuning guardado en: {output_path}")


def plot_sparsity_vs_accuracy(df: pd.DataFrame, output_path: str):
    """
    Grafica relación entre sparsity y accuracy.
    
    Args:
        df: DataFrame con corridas
        output_path: Ruta de salida
    """
    ensure_dir(os.path.dirname(output_path))
    
    # Determinar qué columna de accuracy usar
    acc_col = None
    if 'acc_explore' in df.columns:
        acc_col = 'acc_explore'
    elif 'acc_explore_mean' in df.columns:
        acc_col = 'acc_explore_mean'
    else:
        print(f"    No se encontró columna de accuracy (acc_explore o acc_explore_mean)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter por sparsity
    for sparsity in df['sparsity'].unique():
        df_spar = df[df['sparsity'] == sparsity]
        ax.scatter(df_spar['sparsity'], df_spar[acc_col] * 100,
                  label=f'ρ={sparsity}', alpha=0.6, s=50)
    
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Accuracy exploración (%)')
    ax.set_title('Sparsity vs Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot sparsity guardado en: {output_path}")


def generate_report(df: pd.DataFrame, output_dir: str):
    """
    Genera reporte completo con múltiples visualizaciones.
    
    Args:
        df: DataFrame con corridas
        output_dir: Directorio de salida
    """
    ensure_dir(output_dir)
    
    print(f"\n{'='*60}")
    print("Generando reporte...")
    print(f"{'='*60}")
    
    # Resumen estadístico
    summary = df.describe()
    summary_path = os.path.join(output_dir, 'summary_stats.csv')
    summary.to_csv(summary_path)
    print(f"  Estadísticas guardadas en: {summary_path}")
    
    # Mejores configuraciones (con protección contra columnas faltantes)
    try:
        # Determinar qué columna usar para ranking
        acc_col = None
        if 'acc_explore' in df.columns:
            acc_col = 'acc_explore'
        elif 'acc_explore_mean' in df.columns:
            acc_col = 'acc_explore_mean'
        elif 'acc_explore_median' in df.columns:
            acc_col = 'acc_explore_median'
        
        if acc_col is not None and len(df) > 0:
            best_val = df.nlargest(10, acc_col)
            best_val_path = os.path.join(output_dir, 'top10_val.csv')
            best_val.to_csv(best_val_path, index=False)
            print(f"  Top 10 validación guardadas en: {best_val_path}")
        else:
            print(f"    Columna de accuracy no encontrada (acc_explore, acc_explore_mean, acc_explore_median), saltando top 10")
    except Exception as e:
        print(f"    Error generando top 10: {str(e)}")
    
    # Plots (con protección)
    try:
        if 'sparsity' in df.columns:
            plot_sparsity_vs_accuracy(df, os.path.join(output_dir, 'sparsity_vs_acc.png'))
    except Exception as e:
        print(f"    Error generando plot de sparsity: {str(e)}")
    
    print(f"{'='*60}\n")


class Timer:
    """Contexto para medir tiempo."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = pd.Timestamp.now()
        return self
    
    def __exit__(self, *args):
        self.end_time = pd.Timestamp.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        print(f"{self.name} completado en {elapsed:.2f}s")
    
    def elapsed(self):
        """Retorna tiempo transcurrido."""
        if self.end_time is None:
            return (pd.Timestamp.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()


if __name__ == "__main__":
    print("=== Test: Utilidades ===\n")
    
    # Test Timer
    print("Test: Timer")
    with Timer("Operación de prueba") as t:
        import time
        time.sleep(0.5)
    
    # Test guardar trace ficticio
    print("\nTest: Guardar plot de convergencia")
    
    dummy_trace = {
        'evaluations': list(range(100)),
        'current_history': [0.5 + 0.003*i + 0.01*np.random.randn() for i in range(100)],
        'best_history': [0.5 + 0.003*i for i in range(100)],
        'temperatures': [1.0 * 0.98**i for i in range(50)],
        'acceptance_rates': [0.8 - 0.01*i for i in range(50)]
    }
    
    output_path = "../results/plots/test_convergence.png"
    save_trace_plot(dummy_trace, output_path, title="Test de convergencia")
    
    print("\n Tests completados")
