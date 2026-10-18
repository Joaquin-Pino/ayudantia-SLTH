#!/usr/bin/env python3
"""
Red neuronal densa (MLP totalmente conectada) para Breast Cancer Wisconsin (Original).
UCI Machine Learning Repository, id=15: https://archive.ics.uci.edu/dataset/15

Tarea 1 (ayudantia): construir y evaluar una red densa para este dataset, siguiendo
las convenciones del repo SLTH (CRetamales/SLTH) para que luego el dataset se pueda
integrar al pipeline de busqueda de mascaras sin sorpresas:

  * Mismo split que el repo con train_ratio=0: particion estratificada 75% (trabajo)
    / 25% (test), con seed_split=300. El test se usa UNA sola vez, al final.
  * Misma definicion del MLP (nn.ModuleList de nn.Linear, ReLU en las capas ocultas,
    sin activacion en la salida) y misma inicializacion (He en ocultas, Xavier en la
    salida, biases en cero) con seed_model=100.
  * Mismas cuatro arquitecturas del paper, adaptadas a 9 entradas y 2 clases.
  * Dentro del 75% se hace validacion cruzada estratificada de 10 folds; el
    imputador y el escalador se ajustan SOLO con los folds de entrenamiento.

A diferencia de SLTH, aqui los pesos SI se entrenan (Adam + entropia cruzada): esta
red sirve como referencia "entrenada" para dimensionar la dificultad del dataset.

Uso:
    python3 bcw_dense_nn.py --data breast-cancer-wisconsin.data --out-dir results
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Esquema del dataset (breast-cancer-wisconsin.data, sin encabezado, 11 columnas)
# ----------------------------------------------------------------------------
FEATURES = [
    "clump_thickness",
    "uniformity_cell_size",
    "uniformity_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",  # unica columna con valores faltantes ('?')
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
]
COLUMNS = ["sample_code"] + FEATURES + ["class"]  # class: 2 = benigno, 4 = maligno
N_FEATURES = len(FEATURES)  # 9
N_CLASSES = 2

# Las 4 arquitecturas del paper (Tabla 1), con 9 entradas y 2 salidas.
DEFAULT_ARCHS = [
    [N_FEATURES, 128, N_CLASSES],
    [N_FEATURES, 256, 128, N_CLASSES],
    [N_FEATURES, 512, 256, 128, N_CLASSES],
    [N_FEATURES, 1024, 512, 256, 128, N_CLASSES],
]


# ----------------------------------------------------------------------------
# Semillas (misma logica que src/seeds.py del repo)
# ----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# Modelo (misma estructura que src/model.py del repo, pero entrenable)
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    """Red densa: Linear -> ReLU -> ... -> Linear (logits, sin activacion final)."""

    def __init__(self, architecture: list[int]):
        super().__init__()
        self.architecture = list(architecture)
        self.layers = nn.ModuleList()
        for i in range(len(architecture) - 1):
            self.layers.append(nn.Linear(architecture[i], architecture[i + 1]))
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


def build_mlp(architecture: list[int], seed_model: int) -> MLP:
    """Construye el MLP con la misma inicializacion determinista del repo."""
    set_seed(seed_model)
    model = MLP(architecture)
    for i, layer in enumerate(model.layers):
        if i < len(model.layers) - 1:
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        else:
            nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)
    return model


def count_connections_params(architecture: list[int]) -> tuple[int, int]:
    """Conexiones = pesos entre capas consecutivas; parametros = pesos + biases."""
    conns = sum(a * b for a, b in zip(architecture[:-1], architecture[1:]))
    params = conns + sum(architecture[1:])
    return conns, params


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def load_bcw(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Lee breast-cancer-wisconsin.data y valida su formato.

    Retorna X (n, 9) con NaN donde habia '?', y (n,) con 0 = benigno / 1 = maligno,
    y un dict con estadisticas de calidad de datos.
    """
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")

    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"Se esperaban {len(COLUMNS)} columnas, hay {df.shape[1]}")
    bad_classes = set(df["class"].unique()) - {2, 4}
    if bad_classes:
        raise ValueError(f"Valores de clase inesperados: {sorted(bad_classes)}")
    feats = df[FEATURES]
    if not ((feats.dropna() >= 1) & (feats.dropna() <= 10)).all().all():
        raise ValueError("Hay features fuera del rango [1, 10]")

    X = feats.to_numpy(dtype=float)
    y = (df["class"] == 4).astype(int).to_numpy()

    info = {
        "n_samples": int(len(df)),
        "n_features": N_FEATURES,
        "class_counts": {"benign(2)": int((y == 0).sum()), "malignant(4)": int((y == 1).sum())},
        "missing_by_feature": {k: int(v) for k, v in feats.isna().sum().items() if v > 0},
        "rows_with_missing": int(feats.isna().any(axis=1).sum()),
        "duplicate_sample_codes": int(df["sample_code"].duplicated().sum()),
        "duplicate_feature_label_rows": int(df[FEATURES + ["class"]].duplicated().sum()),
    }
    return X, y, info


def make_preprocessor():
    """Imputacion por mediana + z-score. Siempre se ajusta solo con datos de entrenamiento."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


# ----------------------------------------------------------------------------
# Entrenamiento y evaluacion
# ----------------------------------------------------------------------------
def train_model(
    architecture: list[int],
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed_model: int,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
) -> MLP:
    model = build_mlp(architecture, seed_model)
    model.train()

    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed_model)  # orden de minibatches reproducible

    n = len(yt)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    return model


@torch.no_grad()
def predict(model: MLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(X).float()).argmax(dim=1).numpy()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        # F1 de la clase maligna (la que importa clinicamente) y macro
        "f1_malignant": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def cross_validate(architecture, X_work, y_work, args) -> dict:
    """CV estratificada de 10 folds sobre el conjunto de trabajo (75%)."""
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed_split)
    rows = []
    for tr_idx, va_idx in skf.split(X_work, y_work):
        pre = make_preprocessor()
        X_tr = pre.fit_transform(X_work[tr_idx])  # estadisticas SOLO del fold de entrenamiento
        X_va = pre.transform(X_work[va_idx])
        model = train_model(
            architecture, X_tr, y_work[tr_idx],
            seed_model=args.seed_model, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, weight_decay=args.weight_decay,
        )
        rows.append(metrics(y_work[va_idx], predict(model, X_va)))
    df = pd.DataFrame(rows)
    return {f"cv_{k}_mean": float(df[k].mean()) for k in df} | {f"cv_{k}_std": float(df[k].std(ddof=1)) for k in df}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Ruta a breast-cancer-wisconsin.data (UCI id=15)")
    p.add_argument("--out-dir", default="results", help="Carpeta de salida")
    p.add_argument("--seed-split", type=int, default=300, help="Semilla del split (igual que el repo)")
    p.add_argument("--seed-model", type=int, default=100, help="Semilla de init de pesos (igual que el repo)")
    p.add_argument("--dedup", action="store_true",
                   help="Elimina filas duplicadas exactas (features+clase) ANTES del split, "
                        "para evitar copias de una misma fila en trabajo y test")
    p.add_argument("--test-ratio", type=float, default=0.25)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    # 1) Datos
    X, y, info = load_bcw(args.data)
    print("== Dataset ==")
    print(json.dumps(info, indent=2))

    if args.dedup:
        keep = ~pd.DataFrame(np.column_stack([X, y])).duplicated().to_numpy()  # NaN cuenta como igual a NaN
        print(f"--dedup: {int(keep.sum())} filas unicas de {len(keep)}")
        X, y = X[keep], y[keep]
        info["n_samples_after_dedup"] = int(len(y))
        info["class_counts_after_dedup"] = {"benign(2)": int((y == 0).sum()), "malignant(4)": int((y == 1).sum())}

    # 2) Split estratificado 75/25 (misma llamada que data_loader.py con train_ratio=0)
    set_seed(args.seed_split)
    X_work, X_test, y_work, y_test = train_test_split(
        X, y, test_size=args.test_ratio, stratify=y, random_state=args.seed_split
    )
    majority = float(max((y_test == 0).mean(), (y_test == 1).mean()))
    print(f"\nTrabajo: {len(y_work)} muestras | Test: {len(y_test)} muestras | "
          f"baseline clase mayoritaria en test: {majority:.4f}")

    # 3) Una red por arquitectura
    results = []
    for arch in DEFAULT_ARCHS:
        conns, params = count_connections_params(arch)
        print(f"\n== Arquitectura {arch} | conexiones={conns:,} parametros={params:,} ==")
        t0 = time.time()

        cv = cross_validate(arch, X_work, y_work, args)

        # Modelo final: preprocesador y red ajustados con todo el conjunto de trabajo
        pre = make_preprocessor()
        X_work_s = pre.fit_transform(X_work)
        X_test_s = pre.transform(X_test)
        model = train_model(
            arch, X_work_s, y_work,
            seed_model=args.seed_model, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, weight_decay=args.weight_decay,
        )
        test = {f"test_{k}": v for k, v in metrics(y_test, predict(model, X_test_s)).items()}

        torch.save(model.state_dict(), out_dir / "models" / ("mlp_" + "-".join(map(str, arch)) + ".pt"))
        row = {
            "architecture": "-".join(map(str, arch)),
            "hidden_layers": len(arch) - 2,
            "connections": conns,
            "parameters": params,
            **cv,
            **test,
            "seconds": round(time.time() - t0, 1),
        }
        results.append(row)
        print(f"  CV acc  = {cv['cv_acc_mean']:.4f} +- {cv['cv_acc_std']:.4f}"
              f" | bal_acc = {cv['cv_bal_acc_mean']:.4f}")
        print(f"  TEST acc = {test['test_acc']:.4f} | bal_acc = {test['test_bal_acc']:.4f}"
              f" | F1(maligno) = {test['test_f1_malignant']:.4f}  [{row['seconds']}s]")

    # 4) Guardar resultados (tabla, JSON con configuracion y modelos ya guardados arriba)
    pd.DataFrame(results).to_csv(out_dir / "results.csv", index=False)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "dataset": info, "test_majority_baseline": majority,
                   "results": results}, f, indent=2)
    print(f"\nListo. Resultados en {out_dir.resolve()}")


if __name__ == "__main__":
    main()
