"""
Manejo de máscaras binarias para SLTH.
Funciones para crear, manipular y evaluar máscaras de sub-redes.
"""

import torch
import numpy as np
from typing import List, Tuple
import copy


def create_mask(weight_shapes: List[Tuple[int, int]], 
                sparsity: float, 
                per_layer: bool = True,
                seed: int = None) -> List[torch.Tensor]:
    """
    Crea una máscara binaria inicial con sparsity dado.
    
    Args:
        weight_shapes: Lista de formas de pesos [(out1, in1), (out2, in2), ...]
        sparsity: Proporción de conexiones activas (0.1 = 10% activo)
        per_layer: Si True, respeta sparsity por capa; si False, global
        seed: Semilla para aleatoriedad (opcional)
    
    Returns:
        Lista de tensores binarios (máscaras) por capa
    """
    if seed is not None:
        torch.default_generator.manual_seed(seed)  # solo CPU: evita el costo de torch.manual_seed (siembra CUDA + format_stack)
        np.random.seed(seed)
    
    masks = []
    
    for shape in weight_shapes:
        total_weights = shape[0] * shape[1]
        num_active = int(total_weights * sparsity)
        
        # Crear máscara con num_active conexiones activas
        mask = torch.zeros(total_weights)
        mask[:num_active] = 1.0
        
        # Permutar aleatoriamente
        mask = mask[torch.randperm(total_weights)]
        
        # Reshape a la forma de la capa
        mask = mask.reshape(shape)
        
        masks.append(mask)
    
    return masks


def _repair_layer_connectivity(layer_mask: torch.Tensor) -> torch.Tensor:
    """Repara conectividad mínima fila/columna en una máscara de capa.

    Garantiza que cada neurona tenga al menos una entrada y una salida
    (al menos un 1 por fila y por columna), manteniendo el número total
    de conexiones activas lo más cercano posible al original.
    """
    repaired = layer_mask.clone()
    rows, cols = repaired.shape

    # Versión vectorizada: solo se itera sobre las filas/columnas vacías (pocas),
    # en el mismo orden ascendente que el bucle original, así que consume el RNG
    # igual y produce exactamente el mismo resultado.

    # Una entrada por neurona de salida (fila)
    for i in (repaired.sum(dim=1) == 0).nonzero().flatten().tolist():
        j = torch.randint(0, cols, (1,)).item()
        repaired[i, j] = 1.0

    # Una salida por neurona de entrada (columna); se evalúa tras reparar las filas
    for j in (repaired.sum(dim=0) == 0).nonzero().flatten().tolist():
        i = torch.randint(0, rows, (1,)).item()
        repaired[i, j] = 1.0

    return repaired


def repair_mask_connectivity(mask: List[torch.Tensor], seed: int = None) -> List[torch.Tensor]:
    """Repara una máscara para evitar neuronas completamente desconectadas.

    Criterio de validez (aproximado):
    - En cada capa lineal, cada neurona de salida tiene al menos una entrada activa.
    - Cada neurona de entrada tiene al menos una salida activa.

    Esto garantiza que no haya neuronas totalmente aisladas por capa y
    reduce fuertemente la probabilidad de no tener camino input→output.
    """
    if seed is not None:
        torch.default_generator.manual_seed(seed)  # solo CPU: evita el costo de torch.manual_seed (siembra CUDA + format_stack)
        np.random.seed(seed)

    repaired = []
    for layer_mask in mask:
        if layer_mask.numel() == 0:
            repaired.append(layer_mask.clone())
        else:
            repaired.append(_repair_layer_connectivity(layer_mask))

    return repaired


def ensure_input_output_path(mask: List[torch.Tensor], seed: int = None) -> List[torch.Tensor]:
    """Garantiza al menos un camino completo input→output en la máscara.

    Asume arquitectura secuencial lineal (MLP). Realiza:
    - Elige una neurona "activa" por capa (por índice) propagando desde
      la entrada hasta la salida.
    - En cada capa, enciende la conexión entre la neurona elegida de la
      capa anterior y la neurona elegida de la capa actual.

    Esto fuerza al menos un camino completo, sin intentar optimizar
    sparsity fina (apenas agrega algunas conexiones).
    """
    if seed is not None:
        torch.default_generator.manual_seed(seed)  # solo CPU: evita el costo de torch.manual_seed (siembra CUDA + format_stack)
        np.random.seed(seed)

    if len(mask) == 0:
        return mask

    repaired = [m.clone() for m in mask]

    # Elegir al menos un índice de neurona por capa (columna en primera capa,
    # columna en intermedias y fila en última) y forzar las conexiones.

    # Para la primera capa: elegir neurona de entrada (columna) y de salida (fila)
    first = repaired[0]
    out_first, in_first = first.shape
    in_idx = torch.randint(0, in_first, (1,)).item()
    out_idx = torch.randint(0, out_first, (1,)).item()
    first[out_idx, in_idx] = 1.0

    prev_out_idx = out_idx

    # Capas intermedias (de 2 a L-1): conectar neurona previa con nueva neurona
    for layer_idx in range(1, len(repaired)):
        layer = repaired[layer_idx]
        out_dim, in_dim = layer.shape

        # En capas ocultas, prev_out_idx es un índice de neurona de entrada (columna)
        in_idx = prev_out_idx
        if in_idx >= in_dim:
            in_idx = in_idx % in_dim

        out_idx = torch.randint(0, out_dim, (1,)).item()
        layer[out_idx, in_idx] = 1.0

        prev_out_idx = out_idx

    return repaired


def create_magnitude_mask(model, sparsity: float, per_layer: bool = True) -> List[torch.Tensor]:
    """
    Crea máscara basada en magnitud de pesos (heurística baseline).
    Activa las top-|w| conexiones por magnitud absoluta.
    
    Args:
        model: Modelo MLP con pesos inicializados
        sparsity: Proporción de conexiones a mantener
        per_layer: Si True, aplica por capa; si False, global
    
    Returns:
        Lista de máscaras binarias
    """
    masks = []
    
    with torch.no_grad():
        for layer in model.layers:
            weights = layer.weight.data
            shape = weights.shape
            
            if per_layer:
                # Top-k por capa
                num_active = int(weights.numel() * sparsity)
                threshold = torch.topk(weights.abs().flatten(), num_active, sorted=True)[0][-1]
                mask = (weights.abs() >= threshold).float()
            else:
                # Implementar global si es necesario
                raise NotImplementedError("Global magnitude mask no implementado aún")
            
            masks.append(mask)
    
    return masks


def swap_1to1(mask: List[torch.Tensor], seed: int = None) -> List[torch.Tensor]:
    """
    Genera un vecino mediante swap 1-a-1: apaga una conexión activa, 
    enciende una inactiva. Mantiene exacto el presupuesto de sparsity.
    
    Args:
        mask: Máscara actual (lista de tensores)
        seed: Semilla para reproducibilidad
    
    Returns:
        Nueva máscara (vecino)
    """
    if seed is not None:
        torch.default_generator.manual_seed(seed)  # solo CPU: evita el costo de torch.manual_seed (siembra CUDA + format_stack)
        np.random.seed(seed)
    
    # Copiar máscara
    new_mask = [m.clone() for m in mask]
    
    # Elegir capa aleatoria
    layer_idx = np.random.randint(0, len(new_mask))
    layer_mask = new_mask[layer_idx]
    
    # Obtener índices de activos e inactivos
    flat_mask = layer_mask.flatten()
    active_indices = torch.where(flat_mask == 1.0)[0]
    inactive_indices = torch.where(flat_mask == 0.0)[0]
    
    # Si no hay swap posible, retornar máscara original
    if len(active_indices) == 0 or len(inactive_indices) == 0:
        return new_mask
    
    # Seleccionar índices aleatorios
    off_idx = active_indices[torch.randint(0, len(active_indices), (1,))].item()
    on_idx = inactive_indices[torch.randint(0, len(inactive_indices), (1,))].item()
    
    # Hacer swap
    flat_mask[off_idx] = 0.0
    flat_mask[on_idx] = 1.0
    
    # Reshape y actualizar
    new_mask[layer_idx] = flat_mask.reshape(layer_mask.shape)
    
    return new_mask


def sample_swaps(mask: List[torch.Tensor], K: int, seed: int = None) -> List[Tuple[List[torch.Tensor], Tuple[int, int, int]]]:
    """
    Genera K vecinos mediante swaps 1-a-1.
    Útil para Tabu Search.
    
    Args:
        mask: Máscara actual
        K: Número de vecinos a generar
        seed: Semilla
    
    Returns:
        Lista de tuplas (nueva_máscara, movimiento)
        movimiento = (layer_idx, off_idx, on_idx)
    """
    if seed is not None:
        torch.default_generator.manual_seed(seed)  # solo CPU: evita el costo de torch.manual_seed (siembra CUDA + format_stack)
        np.random.seed(seed)
    
    neighbors = []
    
    for _ in range(K):
        new_mask = [m.clone() for m in mask]
        
        # Elegir capa aleatoria
        layer_idx = np.random.randint(0, len(new_mask))
        layer_mask = new_mask[layer_idx]
        
        flat_mask = layer_mask.flatten()
        active_indices = torch.where(flat_mask == 1.0)[0]
        inactive_indices = torch.where(flat_mask == 0.0)[0]
        
        if len(active_indices) == 0 or len(inactive_indices) == 0:
            continue
        
        off_idx = active_indices[torch.randint(0, len(active_indices), (1,))].item()
        on_idx = inactive_indices[torch.randint(0, len(inactive_indices), (1,))].item()
        
        flat_mask[off_idx] = 0.0
        flat_mask[on_idx] = 1.0
        
        new_mask[layer_idx] = flat_mask.reshape(layer_mask.shape)
        
        move = (layer_idx, off_idx, on_idx)
        neighbors.append((new_mask, move))
    
    return neighbors


def hamming_distance(mask1: List[torch.Tensor], mask2: List[torch.Tensor]) -> int:
    """
    Calcula distancia de Hamming entre dos máscaras.
    Útil para medir diversidad de soluciones.
    
    Args:
        mask1, mask2: Máscaras a comparar
    
    Returns:
        Número de posiciones diferentes
    """
    total_diff = 0
    
    for m1, m2 in zip(mask1, mask2):
        diff = (m1 != m2).sum().item()
        total_diff += diff
    
    return int(total_diff)


def mask_sparsity(mask: List[torch.Tensor]) -> float:
    """
    Calcula la sparsity real de una máscara.
    
    Args:
        mask: Máscara a analizar
    
    Returns:
        Proporción de conexiones activas [0, 1]
    """
    total_params = sum(m.numel() for m in mask)
    active_params = sum((m == 1.0).sum().item() for m in mask)
    
    return active_params / total_params if total_params > 0 else 0.0


def count_active_connections(mask: List[torch.Tensor]) -> int:
    """
    Cuenta el número de conexiones activas en la máscara.
    
    Args:
        mask: Máscara a analizar
    
    Returns:
        Número de conexiones activas
    """
    return sum((m == 1.0).sum().item() for m in mask)


def apply_mask(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Aplica máscara a pesos (elemento a elemento).
    
    Args:
        weights: Tensor de pesos
        mask: Tensor binario de máscara
    
    Returns:
        Pesos enmascarados
    """
    return weights * mask


def masks_equal(mask1: List[torch.Tensor], mask2: List[torch.Tensor]) -> bool:
    """
    Verifica si dos máscaras son idénticas.
    
    Args:
        mask1, mask2: Máscaras a comparar
    
    Returns:
        True si son iguales
    """
    if len(mask1) != len(mask2):
        return False
    
    for m1, m2 in zip(mask1, mask2):
        if not torch.equal(m1, m2):
            return False
    
    return True


def save_mask(mask: List[torch.Tensor], filepath: str):
    """
    Guarda máscara en archivo .pt
    
    Args:
        mask: Máscara a guardar
        filepath: Ruta del archivo
    """
    torch.save(mask, filepath)


def load_mask(filepath: str) -> List[torch.Tensor]:
    """
    Carga máscara desde archivo .pt
    
    Args:
        filepath: Ruta del archivo
    
    Returns:
        Máscara cargada
    """
    return torch.load(filepath, weights_only=False)


if __name__ == "__main__":
    print("=== Test: Manejo de máscaras ===")
    
    # Simular formas de una red Wine (13-128-128-3)
    shapes = [(128, 13), (128, 128), (3, 128)]
    
    # Crear máscara con 20% sparsity
    mask = create_mask(shapes, sparsity=0.2, per_layer=True, seed=42)
    
    print(f"\nMáscara creada con {len(mask)} capas")
    for i, m in enumerate(mask):
        active = (m == 1.0).sum().item()
        total = m.numel()
        print(f"  Capa {i+1}: {active}/{total} activos ({active/total*100:.1f}%)")
    
    print(f"\nSparsity global: {mask_sparsity(mask)*100:.2f}%")
    print(f"Total conexiones activas: {count_active_connections(mask)}")
    
    # Test swap
    print("\n=== Test: Swap 1-a-1 ===")
    mask2 = swap_1to1(mask, seed=123)
    dist = hamming_distance(mask, mask2)
    print(f"Distancia de Hamming después de 1 swap: {dist}")
    print(f"Sparsity después de swap: {mask_sparsity(mask2)*100:.2f}%")
    
    # Test múltiples swaps
    print("\n=== Test: Múltiples swaps ===")
    neighbors = sample_swaps(mask, K=5, seed=456)
    print(f"Generados {len(neighbors)} vecinos")
    for i, (neighbor, move) in enumerate(neighbors):
        dist = hamming_distance(mask, neighbor)
        print(f"  Vecino {i+1}: Hamming = {dist}, Move = capa {move[0]}")
