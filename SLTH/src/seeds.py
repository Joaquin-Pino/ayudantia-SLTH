"""
Manejo de semillas para reproducibilidad.
Controla la aleatoriedad en tres niveles:
- seed_model: inicialización de pesos θ₀
- seed_search: aleatoriedad de la metaheurística
- seed_split: partición de datos
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Fija todas las semillas para reproducibilidad completa.
    
    Args:
        seed: Semilla a establecer
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Asegurar determinismo en operaciones de PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_state():
    """
    Captura el estado actual de los generadores de números aleatorios.
    
    Returns:
        Dict con estados de Python, NumPy y PyTorch
    """
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }


def set_rng_state(state):
    """
    Restaura el estado de los generadores de números aleatorios.
    
    Args:
        state: Dict con estados previamente capturados
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
