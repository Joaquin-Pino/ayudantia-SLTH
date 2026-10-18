"""
Evaluación de sub-redes con máscaras.
Función objetivo para SLTH: accuracy sin entrenamiento.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
import copy


def score(model, mask: List[torch.Tensor], X, y, metric='accuracy') -> float:
    """
    Evalúa una sub-red (modelo + máscara) sin entrenamiento.

    
    Args:
        model: Modelo MLP congelado (debe tener W0 guardado)
        mask: Lista de máscaras binarias por capa
        X: Features (tensor o numpy)
        y: Labels (tensor o numpy)
        metric: Métrica a calcular ('accuracy', 'loss')
    
    Returns:
        Valor de la métrica
    """
    # Convertir a tensores si es necesario
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    if isinstance(y, np.ndarray):
        y = torch.LongTensor(y)
    
    # Modo evaluación
    model.eval()
    
    with torch.no_grad():
        if hasattr(model, 'W0') and model.W0 is not None:
            for layer, W0_layer, mask_layer in zip(model.layers, model.W0, mask):
                layer.weight.data.copy_(W0_layer * mask_layer)
        else:
            # Fallback al comportamiento antiguo (menos eficiente pero funcional)
            original_weights = []
            for layer, mask_layer in zip(model.layers, mask):
                original_weights.append(layer.weight.data.clone())
                layer.weight.data *= mask_layer
        
        # Forward pass
        outputs = model(X)
        
        # Calcular métrica
        if metric == 'accuracy':
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == y).sum().item()
            result = correct / len(y)
        
        elif metric == 'loss':
            criterion = nn.CrossEntropyLoss()
            result = criterion(outputs, y).item()
        
        else:
            raise ValueError(f"Métrica no soportada: {metric}")
        
        if hasattr(model, 'W0') and model.W0 is not None:
            for layer, W0_layer in zip(model.layers, model.W0):
                layer.weight.data.copy_(W0_layer)
        else:
            # Fallback
            for layer, original_w in zip(model.layers, original_weights):
                layer.weight.data = original_w
    
    return result


def evaluate_multiple_metrics(model, mask: List[torch.Tensor], X, y) -> dict:
    """
    Evalúa múltiples métricas simultáneamente.
    
    Args:
        model: Modelo MLP
        mask: Máscara binaria
        X: Features
        y: Labels
    
    Returns:
        Dict con métricas: accuracy, loss, predictions
    """
    # Convertir a tensores
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    if isinstance(y, np.ndarray):
        y = torch.LongTensor(y)
    
    model.eval()
    
    with torch.no_grad():
        if hasattr(model, 'W0') and model.W0 is not None:
            for layer, W0_layer, mask_layer in zip(model.layers, model.W0, mask):
                layer.weight.data.copy_(W0_layer * mask_layer)
        else:
            # Fallback
            original_weights = []
            for layer, mask_layer in zip(model.layers, mask):
                original_weights.append(layer.weight.data.clone())
                layer.weight.data *= mask_layer
        
        # Forward pass
        outputs = model(X)
        predictions = torch.argmax(outputs, dim=1)
        
        # Calcular métricas
        accuracy = (predictions == y).sum().item() / len(y)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, y).item()
        
        # Métricas por clase (si es clasificación multiclase)
        num_classes = outputs.shape[1]
        class_accuracies = []
        for c in range(num_classes):
            mask_c = (y == c)
            if mask_c.sum() > 0:
                acc_c = (predictions[mask_c] == y[mask_c]).sum().item() / mask_c.sum().item()
                class_accuracies.append(acc_c)
            else:
                class_accuracies.append(0.0)
        
        if hasattr(model, 'W0') and model.W0 is not None:
            for layer, W0_layer in zip(model.layers, model.W0):
                layer.weight.data.copy_(W0_layer)
        else:
            for layer, original_w in zip(model.layers, original_weights):
                layer.weight.data = original_w
    
    return {
        'accuracy': accuracy,
        'loss': loss,
        'predictions': predictions.cpu().numpy(),
        'class_accuracies': class_accuracies,
        'num_samples': len(y)
    }


def score_with_mask_stats(model, mask: List[torch.Tensor], X, y) -> Tuple[float, dict]:
    """
    Evalúa y retorna también estadísticas de la máscara.
    
    Args:
        model: Modelo MLP
        mask: Máscara binaria
        X: Features
        y: Labels
    
    Returns:
        Tupla (accuracy, stats_dict)
    """
    from mask import mask_sparsity, count_active_connections
    
    accuracy = score(model, mask, X, y, metric='accuracy')
    
    stats = {
        'accuracy': accuracy,
        'sparsity': mask_sparsity(mask),
        'active_connections': count_active_connections(mask),
        'num_layers': len(mask)
    }
    
    return accuracy, stats


def batch_evaluate(model, masks: List[List[torch.Tensor]], X, y, metric='accuracy') -> List[float]:
    """
    Evalúa múltiples máscaras eficientemente.
    Útil para evaluar población o conjunto de vecinos.
    
    Args:
        model: Modelo MLP
        masks: Lista de máscaras a evaluar
        X: Features
        y: Labels
        metric: Métrica a usar
    
    Returns:
        Lista de scores
    """
    scores = []
    for mask in masks:
        s = score(model, mask, X, y, metric=metric)
        scores.append(s)
    return scores


def compare_masks(model, mask1: List[torch.Tensor], mask2: List[torch.Tensor], 
                  X_val, y_val, X_test, y_test) -> dict:
    """
    Compara dos máscaras en validación y test.
    
    Args:
        model: Modelo MLP
        mask1, mask2: Máscaras a comparar
        X_val, y_val: Datos de validación
        X_test, y_test: Datos de test
    
    Returns:
        Dict con comparación
    """
    from mask import mask_sparsity, hamming_distance
    
    results = {
        'mask1': {
            'acc_explore': score(model, mask1, X_val, y_val),
            'acc_test': score(model, mask1, X_test, y_test),
            'sparsity': mask_sparsity(mask1)
        },
        'mask2': {
            'acc_explore': score(model, mask2, X_val, y_val),
            'acc_test': score(model, mask2, X_test, y_test),
            'sparsity': mask_sparsity(mask2)
        },
        'hamming_distance': hamming_distance(mask1, mask2)
    }
    
    return results


class MaskEvaluator:
    """
    Clase para evaluar máscaras con caché y estadísticas.
    Útil para tracking durante la búsqueda.
    """
    
    def __init__(self, model, X_val, y_val):
        """
        Args:
            model: Modelo MLP congelado
            X_val: Features de validación
            y_val: Labels de validación
        """
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.num_evaluations = 0
        self.best_score = -float('inf')
        self.best_mask = None
        self.evaluation_history = []
    
    def evaluate(self, mask: List[torch.Tensor]) -> float:
        """
        Evalúa una máscara y actualiza estadísticas.
        
        Args:
            mask: Máscara a evaluar
        
        Returns:
            Accuracy
        """
        acc = score(self.model, mask, self.X_val, self.y_val, metric='accuracy')
        
        self.num_evaluations += 1
        self.evaluation_history.append(acc)
        
        if acc > self.best_score:
            self.best_score = acc
            self.best_mask = [m.clone() for m in mask]
        
        return acc
    
    def get_stats(self) -> dict:
        """Obtiene estadísticas de evaluación."""
        return {
            'num_evaluations': self.num_evaluations,
            'best_score': self.best_score,
            'mean_score': np.mean(self.evaluation_history) if self.evaluation_history else 0.0,
            'std_score': np.std(self.evaluation_history) if self.evaluation_history else 0.0
        }
    
    def reset(self):
        """Reinicia estadísticas."""
        self.num_evaluations = 0
        self.best_score = -float('inf')
        self.best_mask = None
        self.evaluation_history = []


if __name__ == "__main__":
    print("=== Test: Evaluación de sub-redes ===")
    
    from model import build_mlp
    from mask import create_mask
    from seeds import set_seed
    
    # Crear modelo simple
    set_seed(42)
    arch = [13, 64, 64, 3]
    model = build_mlp(arch, seed_model=100)
    
    # Datos sintéticos
    np.random.seed(42)
    X = np.random.randn(100, 13)
    y = np.random.randint(0, 3, 100)
    
    # Crear máscaras
    shapes = model.get_weight_shapes()
    mask1 = create_mask(shapes, sparsity=0.3, seed=1)
    mask2 = create_mask(shapes, sparsity=0.3, seed=2)
    
    # Evaluar
    print("\nEvaluando máscara 1...")
    acc1 = score(model, mask1, X, y, metric='accuracy')
    print(f"  Accuracy: {acc1*100:.2f}%")
    
    print("\nEvaluando máscara 2...")
    acc2 = score(model, mask2, X, y, metric='accuracy')
    print(f"  Accuracy: {acc2*100:.2f}%")
    
    # Evaluar múltiples métricas
    print("\n=== Métricas detalladas (máscara 1) ===")
    metrics = evaluate_multiple_metrics(model, mask1, X, y)
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy por clase: {[f'{acc*100:.1f}%' for acc in metrics['class_accuracies']]}")
    
    # Test evaluador con tracking
    print("\n=== Test: MaskEvaluator ===")
    evaluator = MaskEvaluator(model, X, y)
    
    for i in range(5):
        mask = create_mask(shapes, sparsity=0.3, seed=i+10)
        acc = evaluator.evaluate(mask)
        print(f"  Evaluación {i+1}: {acc*100:.2f}%")
    
    stats = evaluator.get_stats()
    print(f"\nEstadísticas finales:")
    print(f"  Evaluaciones: {stats['num_evaluations']}")
    print(f"  Mejor score: {stats['best_score']*100:.2f}%")
    print(f"  Media: {stats['mean_score']*100:.2f}%")
    print(f"  Std: {stats['std_score']*100:.2f}%")
