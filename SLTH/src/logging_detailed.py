"""
Sistema de logging detallado para SLTH.
Registra todas las llamadas a la función objetivo y resultados por estrategia.
"""

import pandas as pd
import numpy as np
import torch
import time
import os
import psutil
from typing import List, Dict, Any
from pathlib import Path


class ObjectiveFunctionLogger:
    """
    Logger para registrar todas las llamadas a la función objetivo.
    
    Registra:
    - Número de llamada
    - Solución (representación de máscara)
    - Neuronas activas
    - Estrategia utilizada
    - Valor de fitness
    - Timestamp
    """
    
    def __init__(self, output_dir: str = "results/logs"):
        """
        Args:
            output_dir: Directorio donde guardar los logs
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.calls = []
        self.call_counter = 0
        
    def log_call(self, 
                 mask: List[torch.Tensor], 
                 fitness: float,
                 strategy: str,
                 iteration: int = None,
                 metadata: Dict[str, Any] = None):
        """
        Registra una llamada a la función objetivo.
        
        Args:
            mask: Máscara binaria (solución)
            fitness: Valor de accuracy/fitness obtenido
            strategy: Nombre de la estrategia (SA, GLS, TABU, etc.)
            iteration: Número de iteración (opcional)
            metadata: Información adicional (opcional)
        """
        self.call_counter += 1
        
        # Representación compacta de la máscara
        mask_repr = self._mask_to_string(mask)
        
        # Contar neuronas activas
        active_neurons = self._count_active_neurons(mask)
        
        # Registro
        call_data = {
            'call_id': self.call_counter,
            'iteration': iteration if iteration is not None else -1,
            'strategy': strategy,
            'solution_repr': mask_repr,
            'active_neurons': active_neurons,
            'fitness': fitness,
            'timestamp': pd.Timestamp.now()
        }
        
        # Agregar metadata si existe
        if metadata:
            call_data.update(metadata)
        
        self.calls.append(call_data)
    
    def _mask_to_string(self, mask: List[torch.Tensor]) -> str:
        """
        Convierte máscara a representación string compacta.
        
        Formato: "L0:10/100,L1:20/200,L2:5/30"
        Indica: Capa 0: 10 activas de 100 totales, etc.
        """
        parts = []
        for layer_idx, mask_layer in enumerate(mask):
            active = int((mask_layer == 1.0).sum().item())
            total = int(mask_layer.numel())
            parts.append(f"L{layer_idx}:{active}/{total}")
        
        return ",".join(parts)
    
    def _count_active_neurons(self, mask: List[torch.Tensor]) -> int:
        """Cuenta el total de conexiones activas en la máscara."""
        total_active = 0
        for mask_layer in mask:
            total_active += int((mask_layer == 1.0).sum().item())
        return total_active
    
    def save_to_csv(self, filename: str):
        """
        Guarda todas las llamadas en CSV.
        
        Args:
            filename: Nombre del archivo o ruta completa
        """
        if len(self.calls) == 0:
            print("[!] No hay llamadas registradas para guardar")
            return
        
        # Si filename es una ruta absoluta o contiene separadores, usarla directamente
        if os.path.isabs(filename) or os.sep in filename or '/' in filename:
            filepath = filename
        else:
            filepath = os.path.join(self.output_dir, filename)
        
        # Asegurar que el directorio padre existe
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            Path(parent_dir).mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.calls)
        
        # Ordenar columnas
        column_order = [
            'call_id', 'iteration', 'strategy', 
            'solution_repr', 'active_neurons', 'fitness', 'timestamp'
        ]
        
        # Agregar columnas adicionales que puedan existir
        other_cols = [col for col in df.columns if col not in column_order]
        column_order.extend(other_cols)
        
        df = df[column_order]
        
        # APPEND MODE: Agregar a archivo existente o crear nuevo
        if os.path.exists(filepath):
            # Cargar datos existentes
            df_existing = pd.read_csv(filepath)
            # Concatenar con nuevos datos
            df_combined = pd.concat([df_existing, df], ignore_index=True)
            df_combined.to_csv(filepath, index=False)
            print(f"Llamadas agregadas a: {filepath}")
            print(f"  Llamadas previas: {len(df_existing)}")
            print(f"  Llamadas nuevas: {len(self.calls)}")
            print(f"  Total acumulado: {len(df_combined)}")
        else:
            df.to_csv(filepath, index=False)
            print(f"Llamadas guardadas en: {filepath}")
            print(f"  Total de llamadas: {len(self.calls)}")
        
        return filepath
    
    def get_statistics(self) -> Dict:
        """Obtiene estadísticas de las llamadas."""
        if len(self.calls) == 0:
            return {}
        
        df = pd.DataFrame(self.calls)
        
        stats = {
            'total_calls': len(self.calls),
            'unique_strategies': df['strategy'].nunique(),
            'strategies': df['strategy'].unique().tolist(),
            'calls_per_strategy': df.groupby('strategy').size().to_dict(),
            'best_fitness': df['fitness'].max(),
            'worst_fitness': df['fitness'].min(),
            'mean_fitness': df['fitness'].mean(),
            'median_fitness': df['fitness'].median()
        }
        
        return stats
    
    def get_summary(self) -> str:
        """Genera un resumen textual de las estadísticas."""
        if len(self.calls) == 0:
            return "No hay llamadas registradas"
        
        stats = self.get_statistics()
        
        summary = f"""
=== Resumen de Llamadas a Función Objetivo ===

Total de llamadas: {stats['total_calls']}
Estrategias únicas: {stats['unique_strategies']}

Llamadas por estrategia:
"""
        for strategy, count in stats['calls_per_strategy'].items():
            pct = (count / stats['total_calls']) * 100
            summary += f"  - {strategy}: {count} ({pct:.1f}%)\n"
        
        summary += f"""
Fitness:
  - Mejor: {stats['best_fitness']:.4f}
  - Peor: {stats['worst_fitness']:.4f}
  - Promedio: {stats['mean_fitness']:.4f}
  - Mediana: {stats['median_fitness']:.4f}
"""
        return summary
    
    def reset(self):
        """Reinicia el contador y limpia los registros."""
        self.calls = []
        self.call_counter = 0


class StrategyResultsLogger:
    """
    Logger para registrar resultados consolidados por estrategia.
    
    Registra:
    - Iteración (asociada a seeds)
    - Código de estrategia
    - Neuronas activadas
    - Solución (representación)
    - Número de llamadas a función objetivo
    - Tiempo de ejecución
    - Memoria utilizada
    """
    
    def __init__(self, output_dir: str = "results/logs"):
        """
        Args:
            output_dir: Directorio donde guardar los logs
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.current_strategy = None
        self.start_time = None
        self.start_memory = None
        self.n_calls = 0
        
    def start_strategy(self, 
                      strategy_code: str,
                      seed_model: int,
                      seed_search: int,
                      iteration: int = None,
                      metadata: Dict[str, Any] = None):
        """
        Inicia el registro de una estrategia.
        
        Args:
            strategy_code: Código de la estrategia (SA, GLS, TABU, etc.)
            seed_model: Semilla del modelo
            seed_search: Semilla de búsqueda
            iteration: Número de iteración/experimento (auto si None)
            metadata: Información adicional
        """
        # Auto-incrementar iteration si no se proporciona
        if iteration is None:
            iteration = len(self.results) + 1
        
        self.current_strategy = {
            'iteration': iteration,
            'strategy_code': strategy_code,
            'seed_model': seed_model,
            'seed_search': seed_search,
            'metadata': metadata or {}
        }
        
        # Timing y memoria
        self.start_time = time.time()
        process = psutil.Process(os.getpid())
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.n_calls = 0
    
    def increment_calls(self, n: int = 1):
        """Incrementa el contador de llamadas a función objetivo."""
        self.n_calls += n
    
    def end_strategy(self, 
                    best_mask: List[torch.Tensor],
                    best_fitness: float,
                    final_stats: Dict[str, Any] = None):
        """
        Finaliza el registro de una estrategia y guarda resultados.
        
        Args:
            best_mask: Mejor máscara encontrada
            best_fitness: Mejor fitness alcanzado
            final_stats: Estadísticas finales adicionales
        """
        if self.current_strategy is None:
            print("[!] No hay estrategia activa para finalizar")
            return
        
        # Calcular tiempo y memoria
        end_time = time.time()
        execution_time = end_time - self.start_time
        
        process = psutil.Process(os.getpid())
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        # Usar máximo entre 0 y la diferencia (evitar negativos por GC)
        memory_usage = max(0, end_memory - self.start_memory)
        
        # Representación de solución
        solution_repr = self._mask_to_string(best_mask)
        active_neurons = self._count_active_neurons(best_mask)
        
        # Compilar resultado
        result = {
            'iteration': self.current_strategy['iteration'],
            'strategy_code': self.current_strategy['strategy_code'],
            'seed_model': self.current_strategy['seed_model'],
            'seed_search': self.current_strategy['seed_search'],
            'active_neurons': active_neurons,
            'solution_repr': solution_repr,
            'best_fitness': best_fitness,
            'n_llamadas_fo': self.n_calls,
            'tiempo_ejecucion_s': execution_time,
            'memoria_ejecucion_mb': memory_usage,
            'timestamp': pd.Timestamp.now()
        }
        
        # Agregar metadata
        result.update(self.current_strategy['metadata'])
        
        # Agregar estadísticas finales si existen
        if final_stats:
            result.update(final_stats)
        
        self.results.append(result)
        
        # Reset
        self.current_strategy = None
        self.start_time = None
        self.start_memory = None
        self.n_calls = 0
    
    def _mask_to_string(self, mask: List[torch.Tensor]) -> str:
        """Convierte máscara a representación string compacta."""
        parts = []
        for layer_idx, mask_layer in enumerate(mask):
            active = int((mask_layer == 1.0).sum().item())
            total = int(mask_layer.numel())
            parts.append(f"L{layer_idx}:{active}/{total}")
        
        return ",".join(parts)
    
    def _count_active_neurons(self, mask: List[torch.Tensor]) -> int:
        """Cuenta el total de conexiones activas."""
        total_active = 0
        for mask_layer in mask:
            total_active += int((mask_layer == 1.0).sum().item())
        return total_active
    
    def save_to_csv(self, filename: str = "resultados_estrategias.csv"):
        """
        Guarda todos los resultados en CSV.
        
        Args:
            filename: Nombre del archivo o ruta completa
        """
        if len(self.results) == 0:
            print("[!] No hay resultados registrados para guardar")
            return
        
        # Si filename es una ruta absoluta o contiene separadores, usarla directamente
        if os.path.isabs(filename) or os.sep in filename or '/' in filename:
            filepath = filename
        else:
            filepath = os.path.join(self.output_dir, filename)
        
        # Asegurar que el directorio padre existe
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            Path(parent_dir).mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.results)
        
        # Ordenar columnas
        column_order = [
            'iteration', 'strategy_code', 'seed_model', 'seed_search',
            'active_neurons', 'solution_repr', 'best_fitness',
            'n_llamadas_fo', 'tiempo_ejecucion_s', 'memoria_ejecucion_mb',
            'timestamp'
        ]
        
        # Agregar columnas adicionales
        other_cols = [col for col in df.columns if col not in column_order]
        column_order.extend(other_cols)
        
        df = df[column_order]
        
        # APPEND MODE: Agregar a archivo existente o crear nuevo
        if os.path.exists(filepath):
            # Cargar datos existentes
            df_existing = pd.read_csv(filepath)
            # Concatenar con nuevos datos
            df_combined = pd.concat([df_existing, df], ignore_index=True)
            df_combined.to_csv(filepath, index=False)
            print(f"Resultados de estrategias agregados a: {filepath}")
            print(f"  Resultados previos: {len(df_existing)}")
            print(f"  Resultados nuevos: {len(self.results)}")
            print(f"  Total acumulado: {len(df_combined)}")
        else:
            df.to_csv(filepath, index=False)
            print(f"Resultados de estrategias guardados en: {filepath}")
            print(f"  Total de estrategias: {len(self.results)}")
        
        return filepath
    
    def get_summary(self) -> Dict:
        """Obtiene resumen de resultados por estrategia."""
        if len(self.results) == 0:
            return {}
        
        df = pd.DataFrame(self.results)
        
        summary = {
            'total_runs': len(self.results),
            'strategies': df['strategy_code'].unique().tolist(),
            'by_strategy': {}
        }
        
        for strategy in df['strategy_code'].unique():
            df_strat = df[df['strategy_code'] == strategy]
            
            summary['by_strategy'][strategy] = {
                'runs': len(df_strat),
                'mean_fitness': df_strat['best_fitness'].mean(),
                'median_fitness': df_strat['best_fitness'].median(),
                'best_fitness': df_strat['best_fitness'].max(),
                'mean_time_s': df_strat['tiempo_ejecucion_s'].mean(),
                'mean_calls': df_strat['n_llamadas_fo'].mean(),
                'mean_memory_mb': df_strat['memoria_ejecucion_mb'].mean()
            }
        
        return summary


def create_combined_loggers(output_dir: str = "results/logs"):
    """
    Crea ambos loggers configurados para trabajar juntos.
    
    Args:
        output_dir: Directorio de salida
    
    Returns:
        Tupla (obj_logger, strategy_logger)
    """
    obj_logger = ObjectiveFunctionLogger(output_dir)
    strategy_logger = StrategyResultsLogger(output_dir)
    
    return obj_logger, strategy_logger


if __name__ == "__main__":
    print("=== Test: Sistema de Logging ===\n")
    
    import torch
    from pathlib import Path
    
    # Crear loggers
    output_dir = "results/test_logs"
    obj_logger, strategy_logger = create_combined_loggers(output_dir)
    
    # Simular experimento
    print("1. Simulando estrategia SA...")
    
    # Máscara dummy
    mask = [
        torch.randint(0, 2, (10, 5)).float(),  # Capa 1
        torch.randint(0, 2, (5, 3)).float()     # Capa 2
    ]
    
    # Iniciar estrategia
    strategy_logger.start_strategy(
        strategy_code="SA",
        iteration=1,
        seed_model=100,
        seed_search=200,
        metadata={'alpha': 0.98, 'T0': 1.0}
    )
    
    # Simular 50 llamadas
    for i in range(50):
        # Modificar máscara ligeramente
        if i > 0:
            # Flip aleatorio
            layer = 0 if i % 2 == 0 else 1
            idx = torch.randint(0, mask[layer].numel(), (1,)).item()
            flat_mask = mask[layer].flatten()
            flat_mask[idx] = 1.0 - flat_mask[idx]
            mask[layer] = flat_mask.reshape(mask[layer].shape)
        
        fitness = 0.5 + 0.01 * i + 0.05 * torch.rand(1).item()
        
        # Log llamada
        obj_logger.log_call(
            mask=mask,
            fitness=fitness,
            strategy="SA",
            iteration=i,
            metadata={'temperature': 1.0 * 0.95**i}
        )
        
        strategy_logger.increment_calls()
        
        # Simular tiempo
        import time
        time.sleep(0.01)
    
    # Finalizar estrategia
    best_fitness = 0.95
    strategy_logger.end_strategy(
        best_mask=mask,
        best_fitness=best_fitness,
        final_stats={'final_temperature': 0.001}
    )
    
    print("\n2. Guardando logs...")
    
    # Guardar
    obj_logger.save_to_csv()
    strategy_logger.save_to_csv()
    
    # Estadísticas
    print("\n3. Estadísticas de llamadas:")
    stats = obj_logger.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n4. Resumen de estrategias:")
    summary = strategy_logger.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nTest completado")
    print(f"\nArchivos generados en: {output_dir}/")
