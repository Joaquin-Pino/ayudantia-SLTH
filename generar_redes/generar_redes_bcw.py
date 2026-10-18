#!/usr/bin/env python3
"""
Genera las redes densas con pesos ALEATORIOS Y CONGELADOS para Breast Cancer
Wisconsin (Original), el punto de partida del enfoque SLTH: los pesos no se entrenan
nunca y solo se buscara una mascara binaria sobre ellos.

Que hace, paso a paso:
  1. Para cada arquitectura (9 entradas, 2 clases) construye el MLP con la misma
     inicializacion determinista del repo SLTH (He en ocultas, Xavier en salida,
     bias en cero) usando seed_model (100 por defecto).
  2. Congela los parametros (requires_grad=False) y deja el modelo en eval().
  3. Guarda los pesos iniciales de cada red en <out-dir>/R*_inicial.pth (state_dict,
     igual que el notebook Genera_Redes) y un resumen en <out-dir>/resumen.csv con
     conexiones y parametros (equivalente a la Tabla 1 del paper).
  4. Si se entrega --data, hace una verificacion de cordura: accuracy de la red
     densa aleatoria SIN mascara sobre el mismo split que usa el repo (75/25
     estratificado, seed_split=300). Se espera algo cercano al azar o a la clase
     mayoritaria, que es justamente lo que la busqueda de mascaras debe mejorar.

Nota: el repo NO carga estos .pth; en cada experimento reconstruye la misma red con
build_mlp(architecture, seed_model). Los archivos sirven de registro y para verificar.

Requiere bcw_dense_nn.py en la misma carpeta (reutiliza su MLP y su loader).

Uso:
    python3 generar_redes_bcw.py --out-dir redes_bcw
    python3 generar_redes_bcw.py --out-dir redes_bcw --data breast-cancer-wisconsin.data [--dedup]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from bcw_dense_nn import (
    DEFAULT_ARCHS,
    build_mlp,
    count_connections_params,
    load_bcw,
    make_preprocessor,
    predict,
    set_seed,
)

# Mismos rotulos que usan los yaml del repo para las 4 arquitecturas (R4 se omite alli).
LABELS = ["R1", "R2", "R3", "R5"]


def freeze(model) -> None:
    """Congela todos los parametros: la red no se entrena nunca (SLTH)."""
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="redes_bcw")
    ap.add_argument("--seed-model", type=int, default=100, help="Semilla de init de pesos (repo: 100)")
    ap.add_argument("--data", default=None, help="breast-cancer-wisconsin.data para la verificacion de cordura")
    ap.add_argument("--dedup", action="store_true", help="Quitar duplicados exactos antes del split (solo con --data)")
    ap.add_argument("--seed-split", type=int, default=300)
    ap.add_argument("--test-ratio", type=float, default=0.25)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Datos para la verificacion de cordura (opcional)
    X_val_s = X_test_s = y_val = y_test = None
    if args.data:
        X, y, _ = load_bcw(args.data)
        if args.dedup:
            keep = ~pd.DataFrame(np.column_stack([X, y])).duplicated().to_numpy()
            X, y = X[keep], y[keep]
            print(f"--dedup: {int(keep.sum())} filas unicas")
        set_seed(args.seed_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X, y, test_size=args.test_ratio, stratify=y, random_state=args.seed_split
        )
        pre = make_preprocessor()
        X_val_s = pre.fit_transform(X_val)  # estadisticas del conjunto de trabajo, como en el repo
        X_test_s = pre.transform(X_test)
        print(f"Split: trabajo={len(y_val)} test={len(y_test)} | "
              f"clase mayoritaria: val={max(y_val.mean(), 1 - y_val.mean()):.4f} "
              f"test={max(y_test.mean(), 1 - y_test.mean()):.4f}")

    rows = []
    for label, arch in zip(LABELS, DEFAULT_ARCHS):
        model = build_mlp(arch, seed_model=args.seed_model)
        freeze(model)

        path = out / f"{label}_inicial.pth"
        torch.save(model.state_dict(), path)

        conns, params = count_connections_params(arch)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == params, f"{label}: conteo de parametros inconsistente"
        assert not any(p.requires_grad for p in model.parameters()), f"{label}: no quedo congelada"

        row = {
            "red": label,
            "arquitectura": "-".join(map(str, arch)),
            "capas_ocultas": len(arch) - 2,
            "conexiones": conns,
            "parametros": params,
            "seed_model": args.seed_model,
            "archivo": path.name,
        }
        if X_val_s is not None:
            row["acc_val_red_densa_aleatoria"] = round(float((predict(model, X_val_s) == y_val).mean()), 4)
            row["acc_test_red_densa_aleatoria"] = round(float((predict(model, X_test_s) == y_test).mean()), 4)
        rows.append(row)
        print(f"{label}  {row['arquitectura']:<22} conexiones={conns:>8,}  parametros={params:>8,}"
              + (f"  acc(val/test, sin mascara)={row['acc_val_red_densa_aleatoria']:.3f}/{row['acc_test_red_densa_aleatoria']:.3f}"
                 if X_val_s is not None else ""))

    pd.DataFrame(rows).to_csv(out / "resumen.csv", index=False)
    print(f"\nGuardado en {out.resolve()}  ({len(rows)} redes + resumen.csv)")


if __name__ == "__main__":
    main()
