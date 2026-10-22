"""
Carga y preprocesamiento de datos (Wine, BCW).
Partición estratificada reproducible.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from seeds import set_seed


def load_wine_data(seed_split: int = 300, 
                   train_ratio: float = 0.6,
                   val_ratio: float = 0.2,
                   test_ratio: float = 0.2) -> Tuple:
    """
    Carga y prepara el dataset Wine con split estratificado reproducible.
    
    Args:
        seed_split: Semilla para el split
        train_ratio: Proporción de datos para entrenamiento
        val_ratio: Proporción para validación
        test_ratio: Proporción para test
    
    Returns:
        Tupla (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    # Verificar proporciones
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Las proporciones deben sumar 1.0"
    
    # Fijar semilla para reproducibilidad
    set_seed(seed_split)
    
    # Cargar datos
    wine = load_wine()
    X = wine.data  # 13 features
    y = wine.target  # 3 clases (0, 1, 2)
    
    print(f"Dataset Wine cargado: {X.shape[0]} muestras, {X.shape[1]} features, {len(np.unique(y))} clases")
    
    if train_ratio == 0.0:
        # Split directo: val vs test (sin train)
        X_train, y_train = np.array([]), np.array([])  # Vacíos
        
        # Split val vs test
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X, y,
            test_size=relative_test_ratio,
            stratify=y,
            random_state=seed_split
        )
        
        # Normalización con estadísticas de val (no hay train)
        scaler = StandardScaler()
        X_val = scaler.fit_transform(X_val)
        X_test = scaler.transform(X_test)
        
    else:
        # Split con train (comportamiento original)
        # Primer split: train vs (val+test)
        temp_ratio = val_ratio + test_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=temp_ratio,
            stratify=y,
            random_state=seed_split
        )
        
        # Segundo split: val vs test
        relative_test_ratio = test_ratio / temp_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=relative_test_ratio,
            stratify=y_temp,
            random_state=seed_split
        )
        
        # Normalización z-score con estadísticas del train
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    
    # Mostrar distribución
    print(f"\nDistribución del split:")
    if train_ratio > 0:
        print(f"  Train: {len(y_train)} muestras ({len(y_train)/len(y)*100:.1f}%)")
    else:
        print(f"  Train: 0 muestras (0.0%) -  SLTH: Sin pre-entrenamiento")
    print(f"  Val:   {len(y_val)} muestras ({len(y_val)/len(y)*100:.1f}%)")
    print(f"  Test:  {len(y_test)} muestras ({len(y_test)/len(y)*100:.1f}%)")
    
    # Distribución por clase
    splits = [('Val', y_val), ('Test', y_test)]
    if train_ratio > 0:
        splits.insert(0, ('Train', y_train))
    
    for split_name, y_split in splits:
        class_counts = np.bincount(y_split)
        print(f"  {split_name} por clase: {class_counts}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def load_bcw_data(seed_split: int = 300,
                  train_ratio: float = 0.0,
                  val_ratio: float = 0.75,
                  test_ratio: float = 0.25,
                  path: str = "data/breast_cancer/breast-cancer-wisconsin.data",
                  dedup: bool = False) -> Tuple:
    """
    Carga Breast Cancer Wisconsin (Original) con el mismo protocolo que Wine.

    Clase 2 -> 0 (benigno), 4 -> 1 (maligno). Los '?' (bare_nuclei) se imputan con
    la mediana del conjunto de trabajo (val), sin fuga hacia test.
    Con dedup=True se quitan duplicados exactos (features + etiqueta) antes del split.

    Returns:
        Tupla (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Las proporciones deben sumar 1.0"

    set_seed(seed_split)

    cols = ["id", "clump_thickness", "uniformity_cell_size", "uniformity_cell_shape",
            "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
            "bland_chromatin", "normal_nucleoli", "mitoses", "class"]
    df = pd.read_csv(path, header=None, names=cols, na_values="?").drop(columns=["id"])
    df["class"] = df["class"].map({2: 0, 4: 1}) ## convertir datos categoricos 
    
    if dedup:
        n0 = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"dedup: {n0} -> {len(df)} filas")

    X = df.drop(columns=["class"]).to_numpy(dtype=np.float64)
    y = df["class"].to_numpy(dtype=np.int64)

    print(f"Dataset BCW cargado: {X.shape[0]} muestras, {X.shape[1]} features, {len(np.unique(y))} clases")

    def _impute(X_fit, *others):
        med = np.nanmedian(X_fit, axis=0)
        out = []
        for A in (X_fit,) + others:
            A = A.copy()
            idx = np.where(np.isnan(A))
            A[idx] = med[idx[1]]
            out.append(A)
        return out

    if train_ratio == 0.0:
        X_train, y_train = np.array([]), np.array([])
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X, y, test_size=relative_test_ratio, stratify=y, random_state=seed_split)
        X_val, X_test = _impute(X_val, X_test)
        scaler = StandardScaler()
        X_val = scaler.fit_transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        temp_ratio = val_ratio + test_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=temp_ratio, stratify=y, random_state=seed_split)
        relative_test_ratio = test_ratio / temp_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=relative_test_ratio, stratify=y_temp, random_state=seed_split)
        X_train, X_val, X_test = _impute(X_train, X_val, X_test)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    print(f"\nDistribución del split: train={len(y_train)} val={len(y_val)} test={len(y_test)}")
    for name, ys in (("Val", y_val), ("Test", y_test)):
        print(f"  {name} por clase: {np.bincount(ys)}")

    return (X_train.astype(np.float32) if len(X_train) else X_train,
            X_val.astype(np.float32), X_test.astype(np.float32),
            y_train, y_val, y_test, scaler)


def load_dataset(name: str, **kwargs) -> Tuple:
    """Despachador de datasets según config['dataset']['name']."""
    if name == "wine":
        kwargs.pop("path", None)
        kwargs.pop("dedup", None)
        return load_wine_data(**kwargs)
    if name == "bcw":
        if kwargs.get("path") is None:
            kwargs.pop("path", None)
        return load_bcw_data(**kwargs)
    raise ValueError(f"Dataset no soportado: {name}")


def load_custom_dataset(filepath: str, 
                       target_column: str,
                       seed_split: int = 300,
                       train_ratio: float = 0.6,
                       val_ratio: float = 0.2,
                       test_ratio: float = 0.2) -> Tuple:
    """
    Carga dataset personalizado desde CSV.
    
    Args:
        filepath: Ruta al archivo CSV
        target_column: Nombre de la columna objetivo
        seed_split: Semilla para split
        train_ratio, val_ratio, test_ratio: Proporciones
    
    Returns:
        Tupla (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    # Fijar semilla
    set_seed(seed_split)
    
    # Cargar CSV
    df = pd.read_csv(filepath)
    
    # Separar features y target
    y = df[target_column].values
    X = df.drop(columns=[target_column]).values
    
    print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} features")
    
    # Split estratificado
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_ratio,
        stratify=y,
        random_state=seed_split
    )
    
    relative_test_ratio = test_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test_ratio,
        stratify=y_temp,
        random_state=seed_split
    )
    
    # Normalización
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def get_data_info(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """
    Obtiene información del dataset.
    
    Returns:
        Dict con estadísticas
    """
    # Manejar caso SLTH (sin train)
    if len(X_train) == 0:
        num_features = X_val.shape[1]
        num_classes = len(np.unique(np.concatenate([y_val, y_test])))
        class_dist_train = []
    else:
        num_features = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        class_dist_train = np.bincount(y_train).tolist()
    
    return {
        'num_features': num_features,
        'num_classes': num_classes,
        'train_size': len(y_train),
        'val_size': len(y_val),
        'test_size': len(y_test),
        'total_size': len(y_train) + len(y_val) + len(y_test),
        'class_distribution_train': class_dist_train,
        'class_distribution_val': np.bincount(y_val).tolist(),
        'class_distribution_test': np.bincount(y_test).tolist()
    }


if __name__ == "__main__":
    print("=== Test: Carga de datos Wine ===\n")
    
    # Cargar con seed fijo
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_wine_data(
        seed_split=300,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    
    # Mostrar estadísticas
    print("\n=== Estadísticas de normalización ===")
    print(f"Media train (primeras 3 features): {X_train[:, :3].mean(axis=0)}")
    print(f"Std train (primeras 3 features): {X_train[:, :3].std(axis=0)}")
    
    # Verificar reproducibilidad
    print("\n=== Test: Reproducibilidad ===")
    X_train2, X_val2, X_test2, y_train2, y_val2, y_test2, _ = load_wine_data(seed_split=300)
    
    print(f"¿Mismo train? {np.allclose(X_train, X_train2) and np.array_equal(y_train, y_train2)}")
    print(f"¿Mismo val? {np.allclose(X_val, X_val2) and np.array_equal(y_val, y_val2)}")
    print(f"¿Mismo test? {np.allclose(X_test, X_test2) and np.array_equal(y_test, y_test2)}")
    
    # Info general
    print("\n=== Información del dataset ===")
    info = get_data_info(X_train, y_train, X_val, y_val, X_test, y_test)
    for key, value in info.items():
        print(f"  {key}: {value}")
