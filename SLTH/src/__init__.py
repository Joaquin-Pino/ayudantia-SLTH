"""
Script para inicialización de __init__.py en el paquete src
"""

__version__ = "1.0.0"
__author__ = "Lab Metaheurísticas"

# Imports principales 
from .seeds import set_seed
from .model import build_mlp, freeze_parameters
from .mask import create_mask, swap_1to1, hamming_distance, mask_sparsity
from .eval import score, MaskEvaluator
from .data_loader import load_wine_data
from .utils import save_trace_plot, Timer

__all__ = [
    'set_seed',
    'build_mlp',
    'freeze_parameters',
    'create_mask',
    'swap_1to1',
    'hamming_distance',
    'mask_sparsity',
    'score',
    'MaskEvaluator',
    'load_wine_data',
    'save_trace_plot',
    'Timer'
]
