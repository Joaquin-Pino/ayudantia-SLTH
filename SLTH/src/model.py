"""
Generador de redes neuronales para SLTH.
Crea MLPs con inicialización determinística y parámetros congelados.
"""

import torch
import torch.nn as nn
from typing import List
import sys
import os

# Permitir import absoluto cuando se ejecuta desde scripts externos
try:
    from .seeds import set_seed
except ImportError:
    from seeds import set_seed


class MLP(nn.Module):
    """
    Multi-Layer Perceptron con arquitectura configurable.
    Diseñado para SLTH: pesos fijos (no entrenables).
    
    FIX CRÍTICO: Guarda copia inmutable de pesos originales (W0)
    para evitar destrucción acumulativa al aplicar máscaras.
    """
    
    def __init__(self, architecture: List[int], activation='relu'):
        """
        Args:
            architecture: Lista con tamaño de cada capa [input, hidden1, ..., output]
            activation: Función de activación ('relu', 'tanh', 'sigmoid')
        """
        super(MLP, self).__init__()
        
        self.architecture = architecture
        self.layers = nn.ModuleList()
        
        # Crear capas lineales
        for i in range(len(architecture) - 1):
            layer = nn.Linear(architecture[i], architecture[i + 1])
            self.layers.append(layer)
        
        self.W0 = None  # Se inicializará después de la inicialización de pesos
        self.b0 = None  # Bias originales
        
        # Seleccionar activación
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Activación no soportada: {activation}")
    
    def forward(self, x):
        """Forward pass sin dropout ni batch-norm para determinismo."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Aplicar activación en todas las capas excepto la última (salida)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x
    
    def get_weight_shapes(self):
        """
        Obtiene las formas de todas las matrices de pesos.
        
        Returns:
            Lista de tuplas (in_features, out_features) por capa
        """
        shapes = []
        for layer in self.layers:
            shapes.append(layer.weight.shape)
        return shapes
    
    def count_parameters(self):
        """Cuenta el número total de parámetros en el modelo."""
        return sum(p.numel() for p in self.parameters())


def build_mlp(architecture: List[int], seed_model: int, activation='relu'):
    """
    Construye un MLP con inicialización determinística.
    
     FIX: Ahora guarda copia de pesos originales (W0) automáticamente.
    
    Args:
        architecture: [input_dim, hidden1, ..., output_dim]
        seed_model: Semilla para inicialización de pesos
        activation: Tipo de activación
    
    Returns:
        Modelo MLP inicializado y congelado (con W0 guardado)
    """
    # Fijar semilla para inicialización
    set_seed(seed_model)
    
    model = MLP(architecture, activation=activation)
    
    # Inicialización He/Kaiming para capas ocultas (óptima para ReLU)
    # Inicialización Xavier para capa de salida
    for i, layer in enumerate(model.layers):
        if i < len(model.layers) - 1:  # Capas ocultas
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        else:  # Capa de salida
            nn.init.xavier_normal_(layer.weight)
        
        # Bias a cero
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    #  CRÍTICO: Guardar copia inmutable de pesos originales
    save_original_weights(model)
    
    # Congelar todos los parámetros (no entrenamiento en SLTH)
    freeze_parameters(model)
    
    # Modo evaluación (desactiva dropout, batch-norm, etc.)
    model.eval()
    
    return model


def freeze_parameters(model):
    """
    Congela todos los parámetros del modelo.
    Esto previene actualizaciones durante backward pass.
    
    Args:
        model: Modelo PyTorch a congelar
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_parameters(model):
    """
    Descongela parámetros (útil si se necesita entrenar después).
    
    Args:
        model: Modelo PyTorch a descongelar
    """
    for param in model.parameters():
        param.requires_grad = True


def save_original_weights(model):
    """
     CRÍTICO: Guarda copia inmutable de pesos originales.
    Debe llamarse UNA VEZ después de inicializar el modelo.
    
    Args:
        model: Modelo MLP recién inicializado
    """
    model.W0 = [layer.weight.data.clone() for layer in model.layers]
    model.b0 = [layer.bias.data.clone() if layer.bias is not None else None 
                for layer in model.layers]


def load_masked_weights(model, mask):
    """
     FIX: Carga pesos enmascarados DESDE LA COPIA ORIGINAL.
    No destruye pesos acumulativamente.
    
    Args:
        model: Modelo MLP (debe tener W0 guardado)
        mask: Lista de máscaras por capa (tensores binarios)
    """
    if model.W0 is None:
        raise ValueError("Model.W0 no inicializado. Llama save_original_weights() primero.")
    
    with torch.no_grad():
        for layer, W0_layer, mask_layer in zip(model.layers, model.W0, mask):
            #  CORRECTO: weight = W0 * mask (NO acumulativo)
            layer.weight.data.copy_(W0_layer * mask_layer)


def apply_mask_to_model(model, mask):
    """
     DEPRECATED: Usa load_masked_weights() en su lugar.
    
    Esta función tiene un BUG CRÍTICO: destruye pesos acumulativamente.
    Se mantiene solo por compatibilidad con código antiguo.
    
    Args:
        model: Modelo MLP
        mask: Lista de máscaras por capa (tensores binarios)
    
    Returns:
        Modelo con pesos enmascarados
    """
    # Usar la versión corregida si W0 existe
    if hasattr(model, 'W0') and model.W0 is not None:
        load_masked_weights(model, mask)
    else:
        # Comportamiento antiguo (BUGGY)
        with torch.no_grad():
            for layer, mask_layer in zip(model.layers, mask):
                layer.weight.data *= mask_layer
    return model


def get_model_summary(model):
    """
    Obtiene un resumen del modelo.
    
    Returns:
        Dict con información del modelo
    """
    total_params = model.count_parameters()
    shapes = model.get_weight_shapes()
    
    return {
        'architecture': model.architecture,
        'total_parameters': total_params,
        'weight_shapes': shapes,
        'num_layers': len(model.layers)
    }


if __name__ == "__main__":
    # Prueba de construcción de modelo
    print("=== Test: Construcción de MLP para Wine ===")
    
    # Arquitectura para Wine: 13 features → 128 → 128 → 3 clases
    arch = [13, 128, 128, 3]
    seed = 100
    
    model = build_mlp(arch, seed_model=seed)
    
    summary = get_model_summary(model)
    print(f"\nArquitectura: {summary['architecture']}")
    print(f"Total parámetros: {summary['total_parameters']:,}")
    print(f"Número de capas: {summary['num_layers']}")
    print(f"\nFormas de pesos por capa:")
    for i, shape in enumerate(summary['weight_shapes']):
        print(f"  Capa {i+1}: {shape}")
    
    # Verificar que está congelado
    print(f"\n¿Parámetros congelados? {not any(p.requires_grad for p in model.parameters())}")
    
    # Test forward pass
    dummy_input = torch.randn(10, 13)  # Batch de 10 muestras
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
