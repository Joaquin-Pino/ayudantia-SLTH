"""
CLI centralizado para generar todos los diagramas y análisis del proyecto SLTH.

Uso:
    python src/generate_diagrams.py convergence       [--results-dir DIR] [--method M] [--top N]
    python src/generate_diagrams.py evolution         [--masks-dir DIR] [--output-dir DIR]
    python src/generate_diagrams.py replicas          <individual.csv> <aggregated.csv> [--output-dir DIR]
    python src/generate_diagrams.py comparison        [--gls-csv CSV] [--opt-csv CSV] [--output-dir DIR]
    python src/generate_diagrams.py stats             [--results-dir DIR] [--target-acc F] [--epsilon F] [--threshold F]
    python src/generate_diagrams.py export-csv        [--results-dir DIR] [--method M] [--top N]
    python src/generate_diagrams.py sparsity          <aggregated.csv> [--individual-csv CSV] [--output-dir DIR]
    python src/generate_diagrams.py diversity         [--masks-dirs DIR [DIR ...]] [--output-dir DIR] [--max-samples N]
    python src/generate_diagrams.py neurons           [--masks-dirs DIR [DIR ...]] [--output-dir DIR]
    python src/generate_diagrams.py boxplots          [--ga-csvs CSV [CSV ...]] [--output-dir DIR]
    python src/generate_diagrams.py ga-evolution      [--masks-dirs DIR [DIR ...]] [--output-dir DIR]
    python src/generate_diagrams.py pareto            [--masks-dir DIR] [--output-dir DIR]
    python src/generate_diagrams.py ss-evolution      [--masks-dir DIR] [--output-dir DIR]
    python src/generate_diagrams.py comparison-table  [--ga-csvs CSV [CSV ...]] [--gls-csv CSV] [--ss-csv CSV] [--output-dir DIR]
    python src/generate_diagrams.py stats-ga          [--ga-csvs CSV [CSV ...]] [--output-dir DIR] [--target-acc F] [--epsilon F] [--threshold F]
    python src/generate_diagrams.py stats-hypervolume [--gls-csv CSV] [--ss-csv CSV] [--ga-dirs DIR [DIR ...]] [--output-dir DIR]
    python src/generate_diagrams.py stats-compare     [--ga-dirs DIR [DIR ...]] [--gls-agg CSV] [--ss-agg CSV] [--output-dir DIR]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ─────────────────────────────────────────────────────────────────────────────
# CONVERGENCIA — desde .pt
# ─────────────────────────────────────────────────────────────────────────────

def _load_pt(pt_path: str) -> dict | None:
    try:
        return torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error cargando {pt_path}: {e}")
        return None


def plot_convergence(trace: dict, title: str, save_path: str):
    """Genera gráfico de convergencia 2×2 desde el trace de un .pt."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    if "accuracy_history" in trace and "evaluations" in trace:
        ax.plot(trace["evaluations"], [a * 100 for a in trace["accuracy_history"]],
                "b-", alpha=0.5, label="Current")
        ax.plot(trace["evaluations"], [a * 100 for a in trace["best_accuracy_history"]],
                "r-", linewidth=2, label="Best")
        ax.set_xlabel("Evaluations"); ax.set_ylabel("Accuracy (%)"); ax.set_title("Convergencia de Accuracy")
        ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if "base_cost_history" in trace and "evaluations" in trace:
        ax.plot(trace["evaluations"], trace["base_cost_history"], "g-", linewidth=1.5, label="Base Cost")
        if "augmented_cost_history" in trace:
            ax.plot(trace["evaluations"], trace["augmented_cost_history"], "orange", alpha=0.7, label="Augmented (GLS)")
        ax.set_xlabel("Evaluations"); ax.set_ylabel("Cost"); ax.set_title("Evolución del Costo")
        ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if "penalty_sum_history" in trace and "evaluations" in trace:
        ax.plot(trace["evaluations"], trace["penalty_sum_history"], "purple", linewidth=1.5)
        ax.set_xlabel("Evaluations"); ax.set_ylabel("Total Penalties"); ax.set_title("Penalizaciones (GLS)")
        ax.grid(True, alpha=0.3)
    elif "temperatures" in trace:
        ax.plot(trace["temperatures"], "m-", linewidth=1.5)
        ax.set_xlabel("Iterations"); ax.set_ylabel("Temperature"); ax.set_title("Enfriamiento (SA)")
        ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "no_improve_counter" in trace and "evaluations" in trace:
        ax.plot(trace["evaluations"], trace["no_improve_counter"], "brown", linewidth=1.5)
        ax.set_xlabel("Evaluations"); ax.set_ylabel("No Improve Counter"); ax.set_title("Estancamiento")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {save_path}")


def cmd_convergence(args):
    """Genera gráficos de convergencia desde archivos .pt."""
    results_dir = Path(args.results_dir)
    method = getattr(args, "method", None)
    top_n = getattr(args, "top", None)

    method_dirs = []
    if method:
        p = results_dir / method
        if p.exists():
            method_dirs.append(p)
    else:
        for sub in results_dir.iterdir():
            if sub.is_dir():
                method_dirs.append(sub)

    all_results = []
    for md in method_dirs:
        masks_dir = md / "best_masks"
        if not masks_dir.exists():
            continue
        for pt_file in masks_dir.glob("*.pt"):
            data = _load_pt(str(pt_file))
            if data and "trace" in data:
                all_results.append({
                    "path": pt_file, "method": md.name,
                    "acc_explore": data.get("acc_explore", 0),
                    "acc_test": data.get("acc_test", 0),
                    "trace": data["trace"],
                })

    if not all_results:
        print("No se encontraron .pt con trace válido.")
        return

    all_results.sort(key=lambda x: x["acc_explore"], reverse=True)
    if top_n:
        all_results = all_results[:top_n]

    plots_dir = Path(args.output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generando {len(all_results)} gráficos de convergencia en {plots_dir} ...")

    for r in all_results:
        filename = r["path"].stem.replace("mask_", "convergence_") + ".png"
        title = (f"{r['method']} - {r['path'].stem}\n"
                 f"Acc Val: {r['acc_explore']*100:.2f}% | Acc Test: {r['acc_test']*100:.2f}%")
        plot_convergence(r["trace"], title, str(plots_dir / filename))

    print(f"Completado: {len(all_results)} gráficos en {plots_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# EVOLUCIÓN PROMEDIO — desde directorio de .pt
# ─────────────────────────────────────────────────────────────────────────────

def cmd_evolution(args):
    """Genera 4 gráficos de evolución promedio desde un directorio de .pt."""
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando traces desde {masks_dir} ...")
    traces = []
    for pt_file in masks_dir.glob("*.pt"):
        data = _load_pt(str(pt_file))
        if not data:
            continue
        trace = data.get("trace", {})
        if "accuracy_history" in trace and len(trace["accuracy_history"]) > 0:
            traces.append({
                "accuracy_history": trace["accuracy_history"],
                "best_accuracy_history": trace["best_accuracy_history"],
                "sparsity": data.get("sparsity", 0.0),
                "acc_test": data.get("acc_test", 0.0),
            })

    if not traces:
        print("No se encontraron traces válidos.")
        return

    print(f"Cargados {len(traces)} traces.")

    max_len = max(len(t["accuracy_history"]) for t in traces)

    def interp_curves(key):
        out = []
        for t in traces:
            h = t[key]
            xo = np.linspace(0, 1, len(h))
            xn = np.linspace(0, 1, max_len)
            out.append(np.interp(xn, xo, h))
        return np.array(out)

    acc_arr = interp_curves("accuracy_history")
    best_arr = interp_curves("best_accuracy_history")
    iters = np.arange(max_len)

    # 1. Evolución promedio global
    fig, ax = plt.subplots(figsize=(16, 8))
    mean_a, std_a = acc_arr.mean(0), acc_arr.std(0)
    mean_b, std_b = best_arr.mean(0), best_arr.std(0)
    ax.plot(iters, mean_a, "b-", linewidth=2, label="Accuracy Actual (promedio)", alpha=0.8)
    ax.fill_between(iters, mean_a - std_a, mean_a + std_a, alpha=0.2, color="blue")
    ax.plot(iters, mean_b, "r-", linewidth=2.5, label="Mejor Accuracy (promedio)", alpha=0.9)
    ax.fill_between(iters, mean_b - std_b, mean_b + std_b, alpha=0.2, color="red")
    ax.axhline(y=mean_b[-1], color="green", linestyle="--", linewidth=1.5,
               label=f"Final: {mean_b[-1]:.2f}%", alpha=0.7)
    ax.set_xlabel("Iteración", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title(f"Evolución Promedio — {len(traces)} Experimentos", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "evolucion_promedio_global.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  1/4 Guardado: {out}")

    # 2. Evolución por sparsity
    sparsities_map: dict = {}
    for t in traces:
        sp = t["sparsity"]
        sparsities_map.setdefault(sp, []).append(t)

    sparsity_list = sorted(sparsities_map)[:4]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Evolución de Accuracy por Sparsity", fontsize=18, fontweight="bold")
    for idx, sp in enumerate(sparsity_list):
        ax = axes[idx // 2, idx % 2]
        sp_traces = sparsities_map[sp]
        ml = max(len(t["best_accuracy_history"]) for t in sp_traces)
        curves = []
        for t in sp_traces:
            h = t["best_accuracy_history"]
            xo = np.linspace(0, 1, len(h)); xn = np.linspace(0, 1, ml)
            curves.append(np.interp(xn, xo, h))
        curves = np.array(curves)
        mean_sp, std_sp = curves.mean(0), curves.std(0)
        its = np.arange(ml)
        ax.plot(its, mean_sp, linewidth=2.5, label=f"Sparsity {sp:.1f}")
        ax.fill_between(its, mean_sp - std_sp, mean_sp + std_sp, alpha=0.3)
        ax.set_title(f"Sparsity {sp:.1f} ({len(sp_traces)} exp.)", fontweight="bold")
        ax.set_xlabel("Iteración", fontweight="bold"); ax.set_ylabel("Mejor Accuracy (%)", fontweight="bold")
        ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    out = output_dir / "evolucion_por_sparsity.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  2/4 Guardado: {out}")

    # 3. Top 10 vs Bottom 10
    sorted_t = sorted(traces, key=lambda t: t["acc_test"], reverse=True)
    top10, bottom10 = sorted_t[:10], sorted_t[-10:]
    fig, ax = plt.subplots(figsize=(16, 8))
    for t in top10:
        ax.plot(t["best_accuracy_history"], "g-", linewidth=1.5, alpha=0.6)
    for t in bottom10:
        ax.plot(t["best_accuracy_history"], "r-", linewidth=1.5, alpha=0.6)

    def _mean_interp(ts):
        ml2 = max(len(t["best_accuracy_history"]) for t in ts)
        cs = []
        for t in ts:
            h = t["best_accuracy_history"]
            xo = np.linspace(0, 1, len(h)); xn = np.linspace(0, 1, ml2)
            cs.append(np.interp(xn, xo, h))
        return np.arange(ml2), np.mean(cs, axis=0)

    ix_t, mt = _mean_interp(top10)
    ix_b, mb = _mean_interp(bottom10)
    ax.plot(ix_t, mt, "g-", linewidth=4,
            label=f"Top 10 Promedio ({top10[0]['acc_test']:.2f}% – {top10[-1]['acc_test']:.2f}%)")
    ax.plot(ix_b, mb, "r-", linewidth=4,
            label=f"Bottom 10 Promedio ({bottom10[0]['acc_test']:.2f}% – {bottom10[-1]['acc_test']:.2f}%)")
    ax.set_xlabel("Iteración", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mejor Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title("Top 10 vs Bottom 10 Experimentos", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "top10_vs_bottom10.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  3/4 Guardado: {out}")

    # 4. Mejora porcentual (convergencia)
    mejoras = []
    for t in traces:
        h = t["best_accuracy_history"]
        if len(h) > 1:
            inicio = h[0]
            xo = np.linspace(0, 1, len(h)); xn = np.linspace(0, 1, 100)
            hi = np.interp(xn, xo, h)
            mejoras.append(((hi - inicio) / (inicio + 1e-10)) * 100)
    mejoras = np.array(mejoras)
    mean_m, std_m = mejoras.mean(0), mejoras.std(0)
    its2 = np.arange(100)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(its2, mean_m, "b-", linewidth=3, label="Mejora Promedio")
    ax.fill_between(its2, mean_m - std_m, mean_m + std_m, alpha=0.3, color="blue")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Progreso de Búsqueda (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mejora desde Inicio (%)", fontsize=14, fontweight="bold")
    ax.set_title("Convergencia: Mejora Porcentual", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    mf = mean_m[-1]
    ax.annotate(f"Mejora final: +{mf:.2f}%", xy=(95, mf), xytext=(70, mf + 5),
                fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    plt.tight_layout()
    out = output_dir / "convergencia_mejora.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  4/4 Guardado: {out}")
    print(f"Completado. Gráficos en: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# RÉPLICAS — convergencia + boxplot por configuración
# ─────────────────────────────────────────────────────────────────────────────

def cmd_replicas(args):
    """Genera convergencia + boxplot para cada configuración de réplicas."""
    df_ind = pd.read_csv(args.individual_csv)
    df_agg = pd.read_csv(args.aggregated_csv)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    configs = df_agg[["architecture", "sparsity", "seed_model"]].drop_duplicates()
    print(f"Generando gráficos para {len(configs)} configuraciones ...")

    for idx, (_, cfg) in enumerate(configs.iterrows(), 1):
        arch, sparsity, seed_model = cfg["architecture"], cfg["sparsity"], cfg["seed_model"]
        print(f"  [{idx}/{len(configs)}] arch={arch}, ρ={sparsity}, seed_model={seed_model}")

        mask_ind = ((df_ind["architecture"] == arch) &
                    (df_ind["sparsity"] == sparsity) &
                    (df_ind["seed_model"] == seed_model))
        replicas = df_ind[mask_ind].copy()
        if len(replicas) == 0:
            continue

        mask_agg = ((df_agg["architecture"] == arch) &
                    (df_agg["sparsity"] == sparsity) &
                    (df_agg["seed_model"] == seed_model))
        if mask_agg.sum() == 0:
            continue
        agg = df_agg[mask_agg].iloc[0].to_dict()

        fig = plt.figure(figsize=(16, 6))

        # Subplot 1: convergencia de la mejor réplica
        ax1 = plt.subplot(1, 2, 1)
        best_idx = replicas["acc_explore"].idxmax()
        best_rep = replicas.loc[best_idx]
        try:
            if "mask_path" in best_rep and pd.notna(best_rep["mask_path"]):
                mp = best_rep["mask_path"]
                if os.path.exists(mp):
                    md = torch.load(mp, weights_only=False)
                    tr = md.get("trace", {})
                    hist = tr.get("best_accuracy_history", tr.get("accuracy_history", []))
                    if hist:
                        ax1.plot(range(len(hist)), [a * 100 for a in hist],
                                 linewidth=2, color="steelblue", label="Mejor réplica")
                        bi = int(np.argmax(hist))
                        ax1.scatter([bi], [hist[bi] * 100], color="red", s=100, zorder=5,
                                    marker="*", label=f"Mejor: {hist[bi]*100:.2f}%")
                        ax1.set_xlabel("Iteración GLS", fontsize=11)
                        ax1.set_ylabel("Accuracy Validación (%)", fontsize=11)
                        ax1.set_title(f"Convergencia — Mejor Réplica\n{arch}, ρ={sparsity}",
                                      fontsize=12, fontweight="bold")
                        ax1.legend(loc="lower right"); ax1.grid(True, alpha=0.3)
        except Exception as e:
            ax1.text(0.5, 0.5, f"Error: {e}", ha="center", va="center",
                     transform=ax1.transAxes, fontsize=9)

        # Subplot 2: boxplot de accuracies
        ax2 = plt.subplot(1, 2, 2)
        acc_explores = replicas["acc_explore"].values * 100
        acc_tests = replicas["acc_test"].values * 100
        bp = ax2.boxplot([acc_explores, acc_tests], labels=["Validación", "Test"],
                         patch_artist=True, widths=0.6,
                         boxprops=dict(facecolor="lightblue", alpha=0.7),
                         medianprops=dict(color="red", linewidth=2),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5))
        for i, data in enumerate([acc_explores, acc_tests], 1):
            x = np.random.normal(i, 0.04, len(data))
            ax2.scatter(x, data, alpha=0.6, s=40, color="darkblue", zorder=3)
        ax2.axhline(y=agg["acc_explore_mean"] * 100, color="green", linestyle="--", linewidth=1.5,
                    alpha=0.7, label=f'Media exploración: {agg["acc_explore_mean"]*100:.2f}%')
        ax2.axhline(y=agg["acc_test_mean"] * 100, color="orange", linestyle="--", linewidth=1.5,
                    alpha=0.7, label=f'Media test: {agg["acc_test_mean"]*100:.2f}%')
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.set_title(f"Distribución ({len(replicas)} réplicas)\n{arch}, ρ={sparsity}",
                      fontsize=12, fontweight="bold")
        ax2.legend(loc="lower right", fontsize=9); ax2.grid(True, axis="y", alpha=0.3)
        stats_text = (f'Val: {agg["acc_explore_mean"]*100:.2f}% ± {agg["acc_explore_std"]*100:.2f}%\n'
                      f'Test: {agg["acc_test_mean"]*100:.2f}% ± {agg["acc_test_std"]*100:.2f}%\n'
                      f'Range: [{agg["acc_explore_min"]*100:.2f}%–{agg["acc_explore_max"]*100:.2f}%]')
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
                 verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        sig = (f"arch_{arch.replace('[','').replace(']','').replace(', ','-')}"
               f"_sp{sparsity}_sm{seed_model}")
        plt.savefig(os.path.join(output_dir, f"replicas_{sig}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Completado. Gráficos en: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPARACIÓN DE DOS MÉTODOS — accuracy, eficiencia, trade-offs, tabla
# ─────────────────────────────────────────────────────────────────────────────

def _calc_stats(df: pd.DataFrame) -> dict:
    return {
        "val_acc_mean": df["acc_explore"].mean(),   "val_acc_std":  df["acc_explore"].std(),
        "val_acc_max":  df["acc_explore"].max(),    "val_acc_min":  df["acc_explore"].min(),
        "test_acc_mean": df["acc_test"].mean(), "test_acc_std": df["acc_test"].std(),
        "gap_val_test": df["acc_explore"].mean() - df["acc_test"].mean(),
        "evals_mean":   df["evaluations"].mean(), "evals_std":  df["evaluations"].std(),
        "time_mean":    df["time_seconds"].mean(), "time_std":  df["time_seconds"].std(),
        "memory_mean":  df["memory_mb"].mean(),   "memory_std": df["memory_mb"].std(),
    }


def cmd_comparison(args):
    """Genera 4 figuras comparativas entre dos métodos (GLS vs GLS_Optimized)."""
    gls_df = pd.read_csv(args.gls_csv)
    opt_df = pd.read_csv(args.opt_csv)
    sg = _calc_stats(gls_df)
    so = _calc_stats(opt_df)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    C_GLS = "#3498db"; C_OPT = "#e74c3c"; C_OK = "#2ecc71"; C_WARN = "#e67e22"
    W = 0.35

    # ── Figura 1: Accuracy ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparación de Accuracy", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    metrics = ["Validation", "Test"]
    gv = [sg["val_acc_mean"] * 100, sg["test_acc_mean"] * 100]
    ov = [so["val_acc_mean"] * 100, so["test_acc_mean"] * 100]
    x = np.arange(2)
    b1 = ax.bar(x - W / 2, gv, W, label="GLS", color=C_GLS, alpha=0.8, edgecolor="black")
    b2 = ax.bar(x + W / 2, ov, W, label="GLS_Opt", color=C_OPT, alpha=0.8, edgecolor="black")
    for bars in [b1, b2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2., b.get_height(),
                    f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Accuracy (%)"); ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title("A. Accuracy Media"); ax.legend(); ax.set_ylim(0, 110); ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    impr = [ov[i] - gv[i] for i in range(2)]
    colors = [C_OK if v >= 0 else C_WARN for v in impr]
    bars = ax.barh(metrics, impr, color=colors, alpha=0.8, edgecolor="black")
    for bar, val in zip(bars, impr):
        lbl = f"+{val:.2f}%" if val >= 0 else f"{val:.2f}%"
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2.,
                f"  {lbl}", va="center", fontweight="bold", fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mejora Absoluta (%)"); ax.set_title("B. Mejora GLS_Opt vs GLS"); ax.grid(axis="x", alpha=0.3)

    ax = axes[1, 0]
    gstd = [sg["val_acc_std"] * 100, sg["test_acc_std"] * 100]
    ostd = [so["val_acc_std"] * 100, so["test_acc_std"] * 100]
    ax.bar(x - W / 2, gstd, W, label="GLS", color=C_GLS, alpha=0.8, edgecolor="black")
    ax.bar(x + W / 2, ostd, W, label="GLS_Opt", color=C_OPT, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Desviación Estándar (%)"); ax.set_xticks(x); ax.set_xticklabels(["Val Acc", "Test Acc"])
    ax.set_title("C. Robustez (variabilidad)"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    cases = ["Mejor Caso\n(Val)", "Peor Caso\n(Val)"]
    gc = [sg["val_acc_max"] * 100, sg["val_acc_min"] * 100]
    oc = [so["val_acc_max"] * 100, so["val_acc_min"] * 100]
    ax.bar(x - W / 2, gc, W, label="GLS", color=C_GLS, alpha=0.8, edgecolor="black")
    ax.bar(x + W / 2, oc, W, label="GLS_Opt", color=C_OPT, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Validation Accuracy (%)"); ax.set_xticks(x); ax.set_xticklabels(cases)
    ax.set_title("D. Casos Extremos"); ax.legend(); ax.set_ylim(0, 115); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  1/4 accuracy_comparison.png")

    # ── Figura 2: Eficiencia ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparación de Eficiencia Computacional", fontsize=15, fontweight="bold")
    meths = ["GLS", "GLS_Opt"]

    ax = axes[0, 0]
    em = [sg["evals_mean"], so["evals_mean"]]; es = [sg["evals_std"], so["evals_std"]]
    ax.bar(meths, em, color=[C_GLS, C_OPT], alpha=0.8, edgecolor="black", yerr=es, capsize=5)
    ax.set_ylabel("Evaluaciones"); ax.set_title("A. Evaluaciones Totales"); ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(meths, es, color=[C_GLS, C_OPT], alpha=0.8, edgecolor="black")
    ax.set_ylabel("Desv. Std."); ax.set_title("B. Consistencia (Std Evaluaciones)"); ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    tm = [sg["time_mean"], so["time_mean"]]; ts = [sg["time_std"], so["time_std"]]
    ax.bar(meths, tm, color=[C_GLS, C_OPT], alpha=0.8, edgecolor="black", yerr=ts, capsize=5)
    ax.set_ylabel("Tiempo (s)"); ax.set_title("C. Tiempo de Ejecución"); ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    mm = [sg["memory_mean"], so["memory_mean"]]; ms = [sg["memory_std"], so["memory_std"]]
    ax.bar(meths, mm, color=[C_GLS, C_OPT], alpha=0.8, edgecolor="black", yerr=ms, capsize=5)
    ax.set_ylabel("Memoria (MB)"); ax.set_title("D. Consumo de Memoria"); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "efficiency_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  2/4 efficiency_comparison.png")

    # ── Figura 3: Trade-offs ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Análisis de Trade-offs", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.scatter(sg["time_mean"], sg["val_acc_mean"] * 100, s=300, c=C_GLS, alpha=0.7,
               edgecolor="black", linewidth=2, label="GLS", marker="o")
    ax.scatter(so["time_mean"], so["val_acc_mean"] * 100, s=300, c=C_OPT, alpha=0.7,
               edgecolor="black", linewidth=2, label="GLS_Opt", marker="s")
    ax.annotate("", xy=(so["time_mean"], so["val_acc_mean"] * 100),
                xytext=(sg["time_mean"], sg["val_acc_mean"] * 100),
                arrowprops=dict(arrowstyle="->", lw=2, color="gray", linestyle="--"))
    ax.set_xlabel("Tiempo (s)"); ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("A. Accuracy vs Tiempo"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    rob_g = 1 / sg["val_acc_std"]; rob_o = 1 / so["val_acc_std"]
    ax.scatter(sg["evals_mean"], rob_g, s=300, c=C_GLS, alpha=0.7,
               edgecolor="black", linewidth=2, label="GLS", marker="o")
    ax.scatter(so["evals_mean"], rob_o, s=300, c=C_OPT, alpha=0.7,
               edgecolor="black", linewidth=2, label="GLS_Opt", marker="s")
    ax.annotate("", xy=(so["evals_mean"], rob_o), xytext=(sg["evals_mean"], rob_g),
                arrowprops=dict(arrowstyle="->", lw=2, color="gray", linestyle="--"))
    ax.set_xlabel("Evaluaciones"); ax.set_ylabel("Robustez (1/Std)")
    ax.set_title("B. Robustez vs Eficiencia"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "tradeoff_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  3/4 tradeoff_analysis.png")

    # ── Figura 4: Tabla resumen ────────────────────────────────────────────────
    def _chg(gv2, ov2, typ="higher"):
        if gv2 == 0:
            return "—"
        if typ == "abs":
            d = ov2 - gv2
            return ("+" if d >= 0 else "") + f"{d*100:.2f}pp"
        pct = (ov2 - gv2) / abs(gv2) * 100
        ok = pct > 0 if typ == "higher" else pct < 0
        sym = "OK" if ok else "!"
        return f"{sym} {pct:+.1f}%"

    rows = [
        ["Val Accuracy (media)", f"{sg['val_acc_mean']*100:.2f}%", f"{so['val_acc_mean']*100:.2f}%", _chg(sg['val_acc_mean'], so['val_acc_mean'])],
        ["Val Accuracy (std)",   f"{sg['val_acc_std']*100:.2f}%",  f"{so['val_acc_std']*100:.2f}%",  _chg(sg['val_acc_std'],  so['val_acc_std'],  "lower")],
        ["Test Accuracy (media)",f"{sg['test_acc_mean']*100:.2f}%",f"{so['test_acc_mean']*100:.2f}%",_chg(sg['test_acc_mean'],so['test_acc_mean'])],
        ["Test Accuracy (std)",  f"{sg['test_acc_std']*100:.2f}%", f"{so['test_acc_std']*100:.2f}%", _chg(sg['test_acc_std'], so['test_acc_std'],  "lower")],
        ["Gap Val-Test",         f"{sg['gap_val_test']*100:.2f}%", f"{so['gap_val_test']*100:.2f}%", _chg(sg['gap_val_test'], so['gap_val_test'],  "lower")],
        ["Evaluaciones (media)", f"{sg['evals_mean']:.0f}",        f"{so['evals_mean']:.0f}",         _chg(sg['evals_mean'],   so['evals_mean'],   "lower")],
        ["Tiempo (media)",       f"{sg['time_mean']:.2f}s",        f"{so['time_mean']:.2f}s",         _chg(sg['time_mean'],    so['time_mean'],    "lower")],
        ["Memoria (media)",      f"{sg['memory_mean']:.2f}MB",     f"{so['memory_mean']:.2f}MB",      _chg(sg['memory_mean'],  so['memory_mean'],  "lower")],
    ]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Métrica", "GLS", "GLS_Opt", "Cambio"],
                   cellLoc="left", loc="center", colWidths=[0.4, 0.2, 0.2, 0.2])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2)
    for j in range(4):
        tbl[(0, j)].set_facecolor("#3498db")
        tbl[(0, j)].set_text_props(weight="bold", color="white")
    for i in range(1, len(rows) + 1):
        if i % 2 == 0:
            for j in range(4):
                tbl[(i, j)].set_facecolor("#f8f9fa")
    plt.title("Tabla Comparativa", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(output_dir / "summary_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  4/4 summary_table.png")
    print(f"Completado. Figuras en: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS ESTADÍSTICOS SLTH
# ─────────────────────────────────────────────────────────────────────────────

def cmd_stats(args):
    """Análisis estadístico completo de la SLTH desde un CSV de resultados."""
    from scipy import stats as scipy_stats

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "runs_individual.csv"
    if not csv_path.exists():
        csv_path = results_dir / "runs.csv"
    if not csv_path.exists():
        print(f"No se encontró CSV en {results_dir}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Cargados {len(df)} runs desde {csv_path}")

    test_acc = df["acc_test"].values
    target = args.target_acc
    epsilon = args.epsilon
    threshold = args.threshold

    output_dir = results_dir / "plots" / "statistical_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shapiro-Wilk
    w_stat, w_pval = scipy_stats.shapiro(test_acc)
    print(f"\nShapiro-Wilk: W={w_stat:.4f}, p={w_pval:.4e} ({'normal' if w_pval>0.05 else 'no normal'})")

    # T-test vs baseline aleatorio (1/3)
    t_stat, t_pval = scipy_stats.ttest_1samp(test_acc, 1 / 3, alternative="greater")
    print(f"T-test vs random: t={t_stat:.4f}, p={t_pval:.4e} ({'sig.' if t_pval<0.05 else 'no sig.'})")

    # TOST vs red objetivo
    t_lo, p_lo = scipy_stats.ttest_1samp(test_acc, target - epsilon, alternative="greater")
    t_hi, p_hi = scipy_stats.ttest_1samp(test_acc, target + epsilon, alternative="less")
    p_tost = max(p_lo, p_hi)
    print(f"TOST (equiv. ε={epsilon:.0%}): p={p_tost:.4e} ({'equiv.' if p_tost<0.05 else 'no equiv.'})")

    # Existencia de winning ticket
    winners = int(np.sum(test_acc >= threshold))
    binom_r = scipy_stats.binomtest(winners, len(test_acc), p=0.0, alternative="greater")
    print(f"Winning ticket (≥{threshold:.0%}): {winners}/{len(test_acc)}, p={binom_r.pvalue:.4e}")

    # IC 95%
    mean_a = test_acc.mean()
    se = scipy_stats.sem(test_acc)
    ci_half = se * scipy_stats.t.ppf(0.975, len(test_acc) - 1)
    print(f"IC 95%: [{mean_a - ci_half:.2%}, {mean_a + ci_half:.2%}]")

    # Gráfico: normalidad
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(test_acc, bins=20, density=True, alpha=0.7, edgecolor="black")
    x_n = np.linspace(test_acc.min(), test_acc.max(), 100)
    ax1.plot(x_n, scipy_stats.norm.pdf(x_n, mean_a, test_acc.std()), "r-", lw=2, label="Normal fit")
    ax1.set_xlabel("Test Accuracy"); ax1.set_ylabel("Density"); ax1.set_title("Histograma + Normal")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    scipy_stats.probplot(test_acc, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "normality_test.png", dpi=300, bbox_inches="tight"); plt.close()

    # Gráfico: IC vs baseline y objetivo
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(0, mean_a, yerr=[[ci_half], [ci_half]], fmt="o", markersize=10,
                capsize=10, capthick=2, color="blue", label="Subnetwork Mean ± 95% CI")
    ax.axhline(1 / 3, color="red", linestyle="--", lw=2, label="Random Baseline (33.3%)")
    ax.axhline(target, color="green", linestyle="--", lw=2, label=f"Target ({target:.0%})")
    ax.axhspan(target - epsilon, target + epsilon, alpha=0.15, color="green",
               label=f"SLTH Tolerance ε={epsilon:.0%}")
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(0, 1); ax.set_ylabel("Test Accuracy")
    ax.set_title("SLTH Validation: Subnetwork vs Target Network")
    ax.set_xticks([]); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.text(0.1, mean_a, f"{mean_a:.2%}", va="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_interval.png", dpi=300, bbox_inches="tight"); plt.close()

    # Gráfico: distribución (boxplot, violin, histograma, CDF)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    bp = ax.boxplot([test_acc], labels=["SLTH Subredes"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    ax.axhline(1 / 3, color="red", linestyle="--", lw=2, label="Random")
    ax.axhline(target, color="green", linestyle="--", lw=2, label=f"Target ({target:.0%})")
    ax.set_ylabel("Test Accuracy"); ax.set_title("Distribution"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.violinplot([test_acc], positions=[0], showmeans=True, showmedians=True)
    ax.axhline(1 / 3, color="red", linestyle="--", lw=2)
    ax.axhline(target, color="green", linestyle="--", lw=2)
    ax.set_ylabel("Test Accuracy"); ax.set_title("Violin Plot")
    ax.set_xticks([0]); ax.set_xticklabels(["SLTH Subredes"]); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(test_acc, bins=20, edgecolor="black", alpha=0.7, color="skyblue")
    ax.axvline(mean_a, color="blue", linestyle="--", lw=2, label=f"Mean: {mean_a:.2%}")
    ax.axvline(1 / 3, color="red", linestyle="--", lw=2, label="Random")
    ax.axvline(target, color="green", linestyle="--", lw=2, label=f"Target: {target:.0%}")
    ax.set_xlabel("Test Accuracy"); ax.set_ylabel("Frequency"); ax.set_title("Histogram")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    sorted_a = np.sort(test_acc)
    cdf = np.arange(1, len(sorted_a) + 1) / len(sorted_a)
    ax.plot(sorted_a, cdf, lw=2, color="blue", label="SLTH Subredes")
    ax.axvline(1 / 3, color="red", linestyle="--", lw=2, label="Random")
    ax.axvline(target, color="green", linestyle="--", lw=2, label=f"Target: {target:.0%}")
    ax.set_xlabel("Test Accuracy"); ax.set_ylabel("CDF"); ax.set_title("Cumulative Distribution")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "distribution_analysis.png", dpi=300, bbox_inches="tight"); plt.close()

    print(f"Completado. Gráficos en: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTAR TRAZAS A CSV
# ─────────────────────────────────────────────────────────────────────────────

def cmd_export_csv(args):
    """Exporta historial de convergencia desde .pt a archivos CSV."""
    results_dir = Path(args.results_dir)
    method = getattr(args, "method", None)
    top_n = getattr(args, "top", None)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for sub in results_dir.iterdir():
        if not sub.is_dir():
            continue
        if method and sub.name != method:
            continue
        masks_dir = sub / "best_masks"
        if not masks_dir.exists():
            continue
        for pt_file in masks_dir.glob("*.pt"):
            data = _load_pt(str(pt_file))
            if data and "trace" in data:
                all_results.append({"path": pt_file, "method": sub.name,
                                    "acc_explore": data.get("acc_explore", 0), "data": data})

    all_results.sort(key=lambda x: x["acc_explore"], reverse=True)
    if top_n:
        all_results = all_results[:top_n]

    print(f"Exportando {len(all_results)} trazas a CSV en {output_dir} ...")
    for r in all_results:
        trace = r["data"]["trace"]
        rows = []
        keys = ["accuracy_history", "best_accuracy_history", "base_cost_history",
                "augmented_cost_history", "penalty_sum_history", "evaluations", "times"]
        length = max((len(trace[k]) for k in keys if k in trace), default=0)
        if length == 0:
            continue
        for i in range(length):
            row = {"method": r["method"], "filename": r["path"].stem,
                   "acc_explore": r["acc_explore"], "step": i}
            for k in keys:
                if k in trace and i < len(trace[k]):
                    row[k] = trace[k][i]
            rows.append(row)
        fname = output_dir / f"history_{r['path'].stem}.csv"
        pd.DataFrame(rows).to_csv(fname, index=False)
        print(f"  Exportado: {fname.name}")

    print(f"Completado. {len(all_results)} CSV en {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY VS SPARSITY + ANÁLISIS DE VARIABILIDAD
# ─────────────────────────────────────────────────────────────────────────────

def cmd_sparsity(args):
    """Genera gráficos de accuracy vs sparsity y análisis de variabilidad."""
    df_agg = pd.read_csv(args.aggregated_csv)
    df_ind = pd.read_csv(args.individual_csv) if args.individual_csv else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resumen por consola
    print(f"\nConfiguraciones totales: {len(df_agg)}")
    print("Top 3 (por acc_explore_mean):")
    for i, (_, row) in enumerate(df_agg.nlargest(3, "acc_explore_mean").iterrows(), 1):
        print(f"  #{i} arch={row['architecture']} ρ={row['sparsity']}: "
              f"val={row['acc_explore_mean']*100:.2f}%±{row['acc_explore_std']*100:.2f}% "
              f"test={row['acc_test_mean']*100:.2f}%")

    # Gráfico 1: accuracy vs sparsity
    grouped = df_agg.groupby("sparsity").agg(
        explore_mean=("acc_explore_mean", "mean"), explore_std=("acc_explore_mean", "std"),
        test_mean=("acc_test_mean", "mean"), test_std=("acc_test_mean", "std")
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.errorbar(grouped["sparsity"], grouped["val_mean"] * 100, yerr=grouped["val_std"] * 100,
                 marker="o", capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.set_xlabel("Sparsity"); ax1.set_ylabel("Accuracy Validación (%)"); ax1.grid(True, alpha=0.3)
    ax1.set_title("Accuracy Validación vs Sparsity")
    ax2.errorbar(grouped["sparsity"], grouped["test_mean"] * 100, yerr=grouped["test_std"] * 100,
                 marker="s", capsize=5, capthick=2, linewidth=2, markersize=8, color="orange")
    ax2.set_xlabel("Sparsity"); ax2.set_ylabel("Accuracy Test (%)"); ax2.grid(True, alpha=0.3)
    ax2.set_title("Accuracy Test vs Sparsity")
    plt.tight_layout()
    out1 = output_dir / "accuracy_by_sparsity.png"
    plt.savefig(out1, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Guardado: {out1}")

    # Gráfico 2: variabilidad top-15
    df_sorted = df_agg.sort_values("acc_explore_std", ascending=False).head(15)
    labels = [f"{r['architecture']}\nρ={r['sparsity']}" for _, r in df_sorted.iterrows()]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(df_sorted)), df_sorted["acc_explore_std"] * 100, color="steelblue", alpha=0.7)
    ax.set_yticks(range(len(df_sorted))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Desviación Estándar de Accuracy (%)"); ax.grid(True, axis="x", alpha=0.3)
    ax.set_title("Top 15 Configuraciones con Mayor Variabilidad")
    plt.tight_layout()
    out2 = output_dir / "variability_analysis.png"
    plt.savefig(out2, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Guardado: {out2}")

    # Gráfico 3 (opcional): distribución de la mejor configuración
    if df_ind is not None:
        best = df_agg.nlargest(1, "acc_explore_mean").iloc[0]
        mask = ((df_ind["architecture"] == best["architecture"]) &
                (df_ind["sparsity"] == best["sparsity"]) &
                (df_ind["seed_model"] == best["seed_model"]))
        reps = df_ind[mask]
        if len(reps) > 0:
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
            a1.hist(reps["acc_explore"] * 100, bins=8, alpha=0.7, color="steelblue", edgecolor="black")
            a1.axvline(best["acc_explore_mean"] * 100, color="red", linestyle="--", lw=2, label="Media")
            a1.set_xlabel("Accuracy Validación (%)"); a1.set_title("Distribución acc_explore"); a1.legend(); a1.grid(True, alpha=0.3)
            a2.hist(reps["acc_test"] * 100, bins=8, alpha=0.7, color="orange", edgecolor="black")
            a2.axvline(best["acc_test_mean"] * 100, color="red", linestyle="--", lw=2, label="Media")
            a2.set_xlabel("Accuracy Test (%)"); a2.set_title("Distribución acc_test"); a2.legend(); a2.grid(True, alpha=0.3)
            plt.tight_layout()
            out3 = output_dir / "best_config_distribution.png"
            plt.savefig(out3, dpi=300, bbox_inches="tight"); plt.close()
            print(f"  Guardado: {out3}")

    print(f"Completado. Gráficos en: {output_dir}")



# ─────────────────────────────────────────────────────────────────────────────
# COLORES DE ARQUITECTURA (compartido por boxplots y otros)
# ─────────────────────────────────────────────────────────────────────────────

ARCHITECTURE_COLORS = {
    '[13, 128, 3]': '#e41a1c',
    '[13, 256, 128, 3]': '#ff7f00',
    '[13, 512, 256, 128, 3]': '#4daf4a',
    '[13, 1024, 512, 256, 128, 3]': '#377eb8',
}


# ─────────────────────────────────────────────────────────────────────────────
# DIVERSITY — análisis de distancia de Hamming entre máscaras
# ─────────────────────────────────────────────────────────────────────────────

def _div_load_mask(pt_path):
    try:
        data = torch.load(pt_path, weights_only=False)
        return {
            'best_mask': data.get('best_mask', []),
            'acc_explore': data.get('acc_explore', 0.0),
            'acc_test': data.get('acc_test', 0.0),
            'architecture': data.get('architecture', []),
            'sparsity': data.get('sparsity', 0.0),
            'path': pt_path,
        }
    except Exception as e:
        print(f"  Error cargando {os.path.basename(pt_path)}: {e}")
        return None


def _div_mask_to_vec(mask_list):
    return np.concatenate([m.cpu().numpy().flatten() for m in mask_list])


def _div_hamming(m1, m2):
    v1, v2 = _div_mask_to_vec(m1), _div_mask_to_vec(m2)
    return np.sum(v1 != v2) / len(v1)


def _div_pairwise(masks_data, max_samples=100):
    from collections import defaultdict
    arch_groups = defaultdict(list)
    for d in masks_data:
        if d:
            arch_groups[str(d['architecture'])].append(d)
    main_arch = max(arch_groups, key=lambda k: len(arch_groups[k]))
    filtered = arch_groups[main_arch]
    if max_samples and len(filtered) > max_samples:
        filtered = sorted(filtered, key=lambda x: x['acc_explore'], reverse=True)
        step = max(1, len(filtered) // max_samples)
        filtered = [filtered[i] for i in range(0, len(filtered), step)][:max_samples]
    n = len(filtered)
    print(f"  Calculando {n*(n-1)//2} distancias de Hamming ({n} máscaras)...")
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _div_hamming(filtered[i]['best_mask'], filtered[j]['best_mask'])
            dist[i, j] = dist[j, i] = d
    meta = [{'acc_explore': d['acc_explore'], 'acc_test': d['acc_test'],
              'sparsity': d['sparsity'], 'path': d['path']} for d in filtered]
    return dist, meta


def _div_metrics(dist_mat):
    n = dist_mat.shape[0]
    ut = dist_mat[np.triu_indices(n, k=1)]
    m = np.mean(ut)
    return {
        'mean_distance': m, 'std_distance': np.std(ut),
        'min_distance': np.min(ut), 'max_distance': np.max(ut),
        'median_distance': np.median(ut),
        'diversity_score': m / 0.5,
        'concentration': np.std(ut) / m if m > 0 else 0,
    }


def _div_plot_distribution(dist_mat, meta, output_path, name):
    from scipy.stats import pearsonr, spearmanr
    n = dist_mat.shape[0]
    dists = dist_mat[np.triu_indices(n, k=1)]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.hist(dists, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(dists), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(dists):.4f}')
    ax.axvline(np.median(dists), color='orange', linestyle='--', linewidth=2, label=f'Mediana: {np.median(dists):.4f}')
    ax.set_xlabel('Distancia de Hamming'); ax.set_ylabel('Frecuencia')
    ax.set_title(f'{name} - Distribución de Distancias'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    bp = ax.boxplot([dists], vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue'); bp['boxes'][0].set_alpha(0.7)
    ax.set_xticklabels([name]); ax.set_ylabel('Distancia de Hamming')
    ax.set_title('Boxplot de Distancias'); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 0]
    sub = dist_mat if n <= 50 else dist_mat[np.ix_(np.linspace(0, n-1, 50, dtype=int),
                                                    np.linspace(0, n-1, 50, dtype=int))]
    im = ax.imshow(sub, cmap='viridis', aspect='auto')
    ax.set_title(f'Matriz de Distancias ({min(n,50)}/{n})'); plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    avg_d = np.mean(dist_mat, axis=1)
    accs = [m['acc_explore'] for m in meta]
    sc = ax.scatter(avg_d, accs, c=accs, cmap='RdYlGn', s=50, alpha=0.6, edgecolors='black')
    plt.colorbar(sc, ax=ax, label='Accuracy')
    r, p = pearsonr(avg_d, accs)
    ax.text(0.05, 0.95, f'Pearson r={r:.3f} (p={p:.3e})', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('Distancia Promedio'); ax.set_ylabel('Val Accuracy')
    ax.set_title('Diversidad vs Calidad'); ax.grid(True, alpha=0.3)

    plt.suptitle(f'{name} - Análisis de Diversidad', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {output_path}")


def _div_plot_landscape(dist_mat, meta, output_path, name):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    accs = np.array([m['acc_explore'] for m in meta])
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, (method, coords) in zip(axes, [
        ('PCA', PCA(n_components=2).fit_transform(dist_mat)),
        ('t-SNE', TSNE(n_components=2, random_state=42,
                       perplexity=min(30, len(meta)-1)).fit_transform(dist_mat)),
    ]):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=accs, cmap='RdYlGn',
                        s=100, alpha=0.7, edgecolors='black', linewidths=1.5)
        best_i = np.argmax(accs)
        ax.scatter(coords[best_i, 0], coords[best_i, 1], marker='*', s=500,
                   c='gold', edgecolors='black', linewidths=2, zorder=10,
                   label=f'Mejor (acc={accs[best_i]:.4f})')
        ax.set_title(f'{name} Fitness Landscape ({method})', fontsize=13, fontweight='bold')
        plt.colorbar(sc, ax=ax, label='Accuracy'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {output_path}")


def _div_plot_dendrogram(dist_mat, output_path, name):
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import dendrogram, linkage
    condensed = squareform(dist_mat)
    Z = linkage(condensed, method='average')
    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(Z, ax=ax, color_threshold=0.7 * max(Z[:, 2]), above_threshold_color='gray')
    thresh = 0.7 * max(Z[:, 2])
    ax.axhline(y=thresh, color='red', linestyle='--', linewidth=2, label=f'Threshold: {thresh:.4f}')
    ax.set_xlabel('Máscara ID'); ax.set_ylabel('Distancia de Hamming')
    ax.set_title(f'{name} - Clustering Jerárquico', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {output_path}")


def _div_write_report(metrics, output_path, name):
    score = metrics['diversity_score']
    if score > 1.2:
        interp = "ALTA - Buena exploración del espacio"
    elif score > 0.8:
        interp = "BUENA - Balance exploración/explotación"
    elif score > 0.5:
        interp = "MODERADA - Cierta convergencia"
    else:
        interp = "BAJA - Convergencia prematura"
    lines = [
        "=" * 70, f"REPORTE DE DIVERSIDAD - {name}", "=" * 70,
        f"Distancia Promedio:  {metrics['mean_distance']:.4f}",
        f"Std:                 {metrics['std_distance']:.4f}",
        f"Min/Max:             [{metrics['min_distance']:.4f}, {metrics['max_distance']:.4f}]",
        f"Diversity Score:     {score:.2f} → {interp}",
        f"Concentration (CV):  {metrics['concentration']:.2f}",
        "=" * 70,
    ]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Guardado: {output_path}")


def _div_compare(results, output_dir):
    if len(results) < 2:
        return
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for r in results:
        n = r['dist_mat'].shape[0]
        dists = r['dist_mat'][np.triu_indices(n, k=1)]
        ax.hist(dists, bins=30, alpha=0.5, label=r['name'], edgecolor='black')
    ax.set_xlabel('Distancia de Hamming'); ax.set_ylabel('Frecuencia')
    ax.set_title('Comparación de Distribuciones'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    data = []
    labels = []
    for r in results:
        n = r['dist_mat'].shape[0]
        data.append(r['dist_mat'][np.triu_indices(n, k=1)])
        labels.append(r['name'])
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral', 'lightgreen']):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylabel('Distancia de Hamming'); ax.set_title('Comparación de Diversidad')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'diversity_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  Guardado: {path}")


def cmd_diversity(args):
    """Análisis de diversidad de máscaras via distancia de Hamming."""
    import glob as _glob
    masks_dirs = args.masks_dirs
    output_dir = Path(args.output_dir)
    max_samples = args.max_samples

    results = []
    for masks_dir in masks_dirs:
        name = Path(masks_dir).parent.name or Path(masks_dir).name
        out = output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        pt_files = _glob.glob(os.path.join(masks_dir, "mask_*.pt"))
        if not pt_files:
            print(f"  Sin archivos .pt en {masks_dir}")
            continue
        masks_data = [_div_load_mask(f) for f in pt_files]
        masks_data = [m for m in masks_data if m]
        if len(masks_data) < 2:
            print(f"  Menos de 2 máscaras válidas en {masks_dir}")
            continue
        dist_mat, meta = _div_pairwise(masks_data, max_samples=max_samples)
        metrics = _div_metrics(dist_mat)
        _div_plot_distribution(dist_mat, meta, str(out / 'distance_distribution.png'), name)
        _div_plot_landscape(dist_mat, meta, str(out / 'fitness_landscape_2d.png'), name)
        _div_plot_dendrogram(dist_mat, str(out / 'clustering_dendrogram.png'), name)
        _div_write_report(metrics, str(out / 'diversity_report.txt'), name)
        results.append({'name': name, 'metrics': metrics, 'dist_mat': dist_mat, 'meta': meta})

    if len(results) >= 2:
        _div_compare(results, str(output_dir / 'comparison'))

    print(f"Completado. Resultados en: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# NEURONS — análisis de neuronas activas/inactivas por capa
# ─────────────────────────────────────────────────────────────────────────────

def _neu_load_pt(pt_path):
    data = torch.load(pt_path, weights_only=False)
    return {
        'mask': data.get('best_mask', []),
        'architecture': data.get('architecture', 'unknown'),
        'sparsity': data.get('sparsity', 0.0),
        'seed_model': data.get('seed_model', 0),
        'seed_search': data.get('seed_search', 0),
        'acc_explore': data.get('acc_explore', 0.0),
        'acc_test': data.get('acc_test', 0.0),
    }


def _neu_analyze_layers(mask_list):
    if isinstance(mask_list, dict):
        mask_list = list(mask_list.values())
    layer_names = [f'fc{i//2+1}.{"weight" if i%2==0 else "bias"}' for i in range(len(mask_list))]
    layer_info = {}
    for idx, tensor in enumerate(mask_list):
        if not isinstance(tensor, torch.Tensor):
            continue
        m = tensor.cpu().numpy()
        name = layer_names[idx] if idx < len(layer_names) else f'layer_{idx}'
        if m.ndim == 2:
            act_out = (m.sum(axis=1) > 0).astype(int)
            act_in  = (m.sum(axis=0) > 0).astype(int)
            tot = m.size
            act_conn = int(m.sum())
            layer_info[name] = {
                'type': 'weight', 'shape': m.shape,
                'output_neurons_total': m.shape[0],
                'output_neurons_active': int(act_out.sum()),
                'output_neurons_inactive': m.shape[0] - int(act_out.sum()),
                'output_neurons_inactive_list': np.where(act_out == 0)[0].tolist(),
                'input_neurons_total': m.shape[1],
                'input_neurons_active': int(act_in.sum()),
                'input_neurons_inactive': m.shape[1] - int(act_in.sum()),
                'total_connections': tot,
                'active_connections': act_conn,
                'inactive_connections': tot - act_conn,
                'sparsity': 1.0 - act_conn / tot,
                'connections_per_output_neuron': m.sum(axis=1).tolist(),
                'connections_per_input_neuron': m.sum(axis=0).tolist(),
            }
        elif m.ndim == 1:
            act = (m > 0).astype(int)
            layer_info[name] = {
                'type': 'bias', 'shape': m.shape,
                'total_biases': m.shape[0], 'active_biases': int(act.sum()),
            }
    return layer_info


def _neu_plot_neurons(layer_info, output_path, title_suffix=""):
    w_layers = {k: v for k, v in layer_info.items() if v['type'] == 'weight'}
    if not w_layers:
        return
    n = len(w_layers)
    fig, axes = plt.subplots(n, 3, figsize=(20, 6*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    colors = ['#2ecc71', '#e74c3c']
    for idx, (lname, info) in enumerate(w_layers.items()):
        for col, (vals, tot, ylabel, title_part) in enumerate([
            ([info['output_neurons_active'], info['output_neurons_inactive']],
             info['output_neurons_total'], 'Neuronas', 'Salida'),
            ([info['input_neurons_active'], info['input_neurons_inactive']],
             info['input_neurons_total'], 'Neuronas', 'Entrada'),
            ([info['active_connections'], info['inactive_connections']],
             info['total_connections'], 'Conexiones', f"Sparsity {info['sparsity']:.2%}"),
        ]):
            ax = axes[idx, col]
            bars = ax.bar(['Activas', 'Inactivas'], vals, color=colors, alpha=0.7, edgecolor='black')
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{v:,}\n({v/tot*100:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(f'{lname} - {title_part}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle(f'Análisis de Neuronas\n{title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {output_path}")


def _neu_plot_heatmap(layer_info, output_path, title_suffix=""):
    w_layers = {k: v for k, v in layer_info.items() if v['type'] == 'weight'}
    if not w_layers:
        return
    n = len(w_layers)
    fig, axes = plt.subplots(1, n, figsize=(8*n, 6))
    if n == 1:
        axes = [axes]
    for ax, (lname, info) in zip(axes, w_layers.items()):
        conns = np.array(info['connections_per_output_neuron']).reshape(-1, 1)
        im = ax.imshow(conns, cmap='YlOrRd', aspect='auto')
        for ni in info['output_neurons_inactive_list']:
            ax.axhline(y=ni, color='blue', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Neurona de Salida', fontweight='bold')
        ax.set_title(f'{lname}\nConexiones por Neurona', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='N° Conexiones'); ax.set_xticks([])
    plt.suptitle(f'Heatmap - {title_suffix}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {output_path}")


def _neu_export_csv(all_info, output_path):
    rows = []
    for mi in all_info:
        for lname, ld in mi['layer_info'].items():
            if ld['type'] == 'weight':
                rows.append({
                    'architecture': str(mi['architecture']),
                    'seed_model': mi['seed_model'], 'seed_search': mi['seed_search'],
                    'acc_explore': mi['acc_explore'], 'acc_test': mi['acc_test'],
                    'layer': lname,
                    'output_neurons_total': ld['output_neurons_total'],
                    'output_neurons_active': ld['output_neurons_active'],
                    'output_active_pct': ld['output_neurons_active'] / ld['output_neurons_total'] * 100,
                    'input_neurons_total': ld['input_neurons_total'],
                    'input_neurons_active': ld['input_neurons_active'],
                    'active_connections': ld['active_connections'],
                    'sparsity': ld['sparsity'],
                })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Guardado: {output_path}")


def cmd_neurons(args):
    """Análisis de neuronas activas/inactivas por capa desde máscaras .pt."""
    import glob as _glob
    masks_dirs = args.masks_dirs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_info = []
    for i, masks_dir in enumerate(masks_dirs):
        replica_name = f"R{i+1}"
        pt_files = _glob.glob(os.path.join(masks_dir, "mask_*.pt"))
        print(f"  {replica_name} ({masks_dir}): {len(pt_files)} archivos")
        for pt_file in pt_files:
            try:
                md = _neu_load_pt(pt_file)
                layer_info = _neu_analyze_layers(md['mask'])
                entry = {'file': os.path.basename(pt_file), **{k: md[k] for k in
                         ['architecture', 'sparsity', 'seed_model', 'seed_search', 'acc_explore', 'acc_test']},
                         'layer_info': layer_info}
                all_info.append(entry)
            except Exception as e:
                print(f"    Error en {os.path.basename(pt_file)}: {e}")

    if not all_info:
        print("No se encontraron máscaras.")
        return

    print(f"\n  Total máscaras analizadas: {len(all_info)}")

    for mi in all_info[:3]:
        stem = mi['file'].replace('.pt', '')
        _neu_plot_neurons(mi['layer_info'], str(output_dir / f"neurons_{stem}.png"),
                          title_suffix=f"{stem} acc_test={mi['acc_test']:.4f}")
        _neu_plot_heatmap(mi['layer_info'], str(output_dir / f"heatmap_{stem}.png"),
                          title_suffix=stem)

    _neu_export_csv(all_info, str(output_dir / 'neurons_detailed_analysis.csv'))

    # Reporte de resumen
    from collections import defaultdict
    by_arch = defaultdict(list)
    for mi in all_info:
        by_arch[str(mi['architecture'])].append(mi)
    report = [f"Total máscaras: {len(all_info)}", f"Arquitecturas: {len(by_arch)}"]
    for arch, items in sorted(by_arch.items()):
        report.append(f"\nArquitectura: {arch} ({len(items)} masks)")
        layer_stats = defaultdict(lambda: {'out_act': [], 'sparsity': []})
        for mi in items:
            for ln, ld in mi['layer_info'].items():
                if ld['type'] == 'weight':
                    layer_stats[ln]['out_act'].append(ld['output_neurons_active'])
                    layer_stats[ln]['sparsity'].append(ld['sparsity'])
        for ln, st in sorted(layer_stats.items()):
            report.append(f"  {ln}: out_active={np.mean(st['out_act']):.1f}±{np.std(st['out_act']):.1f}, "
                          f"sparsity={np.mean(st['sparsity']):.2%}±{np.std(st['sparsity']):.2%}")
    rpath = output_dir / 'neurons_summary_report.txt'
    rpath.write_text('\n'.join(report), encoding='utf-8')
    print(f"  Guardado: {rpath}")
    print(f"Completado. Resultados en: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# BOXPLOTS — boxplots unificados por arquitectura (GA+TABU+NSGA2)
# ─────────────────────────────────────────────────────────────────────────────

def _bp_load_data(csv_paths):
    dfs = []
    for p in csv_paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
        else:
            print(f"  No encontrado: {p}")
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(subset=['architecture', 'sparsity', 'seed_model', 'seed_search', 'replica_id'],
                       inplace=True)
    df['architecture'] = df['architecture'].astype(str)
    return df


def _bp_create_boxplot(df, y_col, title, ylabel, output_path, arch_order, ylim=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    data = [df[df['architecture'] == a][y_col].values for a in arch_order]
    bp = ax.boxplot(data, positions=range(len(arch_order)), widths=0.6, patch_artist=True,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='darkred', markersize=8),
                    medianprops=dict(color='black', linewidth=2.5))
    for patch, arch in zip(bp['boxes'], arch_order):
        patch.set_facecolor(ARCHITECTURE_COLORS.get(arch, '#999999')); patch.set_alpha(0.7)
    labels = [a.replace('[', '').replace(']', '').replace(' ', '') for a in arch_order]
    ax.set_xticks(range(len(arch_order))); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=11)
    ax.set_xlabel('Arquitectura', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    if ylim:
        ax.set_ylim(ylim)
    ax.axhline(y=df[y_col].median(), color='red', linestyle='--', linewidth=1.5, alpha=0.5,
               label=f'Mediana global: {df[y_col].median():.4f}')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y'); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  Guardado: {output_path}")


def cmd_boxplots(args):
    """Boxplots unificados por arquitectura para GA+TABU+NSGA2."""
    df = _bp_load_data(args.ga_csvs)
    if df is None:
        print("No se pudieron cargar datos.")
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arch_order = sorted(df['architecture'].unique(),
                        key=lambda x: sum(map(int, x.strip('[]').split(', '))))
    print(f"  {len(df)} registros únicos, {len(arch_order)} arquitecturas")

    _bp_create_boxplot(df, 'acc_explore', 'GA+Tabu+NSGA-II: Accuracy de Validación por Arquitectura',
                       'Accuracy (Validación)', str(output_dir / 'boxplot_1_accuracy.png'), arch_order, ylim=(0, 1.05))
    _bp_create_boxplot(df, 'evaluations', 'GA+Tabu+NSGA-II: Evaluaciones por Arquitectura',
                       'Número de Evaluaciones', str(output_dir / 'boxplot_2_evaluations.png'), arch_order)
    _bp_create_boxplot(df, 'time_seconds', 'GA+Tabu+NSGA-II: Tiempo por Arquitectura',
                       'Tiempo (segundos)', str(output_dir / 'boxplot_3_time.png'), arch_order)
    if 'final_hv' in df.columns:
        _bp_create_boxplot(df, 'final_hv', 'GA+Tabu+NSGA-II: Hipervolumen por Arquitectura',
                           'Hipervolumen', str(output_dir / 'boxplot_4_hypervolume.png'), arch_order)
    if 'memory_mb' in df.columns:
        _bp_create_boxplot(df, 'memory_mb', 'GA+Tabu+NSGA-II: Memoria por Arquitectura',
                           'Memoria (MB)', str(output_dir / 'boxplot_5_memory.png'), arch_order)

    stats_rows = []
    for arch in arch_order:
        da = df[df['architecture'] == arch]
        row = {'Arquitectura': arch, 'N_runs': len(da),
               'Acc_val_mean': da['acc_explore'].mean(), 'Acc_val_std': da['acc_explore'].std(),
               'Acc_val_median': da['acc_explore'].median(),
               'Evaluations_mean': da['evaluations'].mean(),
               'Time_mean': da['time_seconds'].mean()}
        if 'final_hv' in da.columns:
            row['HV_mean'] = da['final_hv'].mean()
        stats_rows.append(row)
    pd.DataFrame(stats_rows).to_csv(output_dir / 'summary_stats.csv', index=False)
    print(f"  Guardado: {output_dir / 'summary_stats.csv'}")
    print(f"Completado. Resultados en: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# GA-EVOLUTION — evolución de accuracy multi-objetivo (GA+TABU+NSGA2)
# ─────────────────────────────────────────────────────────────────────────────

def _gaevo_load_trace(pt_path):
    data = torch.load(pt_path, weights_only=False)
    trace = data.get('trace', {})
    return {
        'best_accuracy_history': trace.get('best_accuracy_history', []),
        'median_f1_history': trace.get('median_f1_history', []),
        'pareto_front_history': trace.get('pareto_front_history', []),
        'final_stats': trace.get('final_stats', {}),
        'architecture': data.get('architecture', 'unknown'),
        'sparsity': data.get('sparsity', 0.0),
        'acc_explore': data.get('acc_explore', 0.0),
        'acc_test': data.get('acc_test', 0.0),
    }


def _gaevo_plot_single(trace, output_path, title_suffix=""):
    best = trace['best_accuracy_history']
    f1h  = trace['median_f1_history']
    ph   = trace['pareto_front_history']
    gens = list(range(1, len(best) + 1))
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))

    ax = axes[0]
    ax.plot(gens, best, 'o-', color='darkgreen', linewidth=2.5, markersize=5,
            label='Mejor Accuracy (Validación)')
    fs = trace['final_stats']
    ax.text(0.02, 0.98, f"Final: {best[-1] if best else 0:.4f}\nEvals: {fs.get('total_evaluations',0):,}",
            transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5), fontsize=10)
    ax.set_xlabel('Generación'); ax.set_ylabel('Accuracy')
    ax.set_title(f'Evolución Accuracy - {title_suffix}', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05]); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(gens, f1h, 's-', color='steelblue', linewidth=2, markersize=5,
            label='F1 Mediano (Frente Pareto)')
    ax.set_xlabel('Generación'); ax.set_ylabel('F1 Score')
    ax.set_title('Evolución F1 Mediano', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.05]); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    sizes = [len(f) for f in ph]
    ax.plot(gens, sizes, '^-', color='purple', linewidth=2, markersize=5,
            label='Tamaño Frente de Pareto')
    ax.set_xlabel('Generación'); ax.set_ylabel('N° Soluciones')
    ax.set_title('Evolución del Frente de Pareto', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  Guardado: {output_path}")


def _gaevo_plot_aggregated(traces, output_path, title_suffix="", group_label=""):
    max_len = max(len(t['best_accuracy_history']) for t in traces)
    best_mat = np.full((len(traces), max_len), np.nan)
    f1_mat   = np.full((len(traces), max_len), np.nan)
    for i, t in enumerate(traces):
        bh = t['best_accuracy_history']; best_mat[i, :len(bh)] = bh
        fh = t['median_f1_history'];     f1_mat[i, :len(fh)]   = fh
    bm, bs = np.nanmean(best_mat, 0), np.nanstd(best_mat, 0)
    fm, fs2 = np.nanmean(f1_mat, 0),  np.nanstd(f1_mat, 0)
    gens = list(range(1, max_len + 1))
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))

    ax = axes[0]
    ax.fill_between(gens, bm - bs, bm + bs, alpha=0.2, color='darkgreen')
    ax.plot(gens, bm, 'o-', color='darkgreen', linewidth=2.5, markersize=6, label=f'Promedio (n={len(traces)})')
    ax.set_xlabel('Generación'); ax.set_ylabel('Accuracy')
    ax.set_title(f'Evolución Promedio Accuracy - {title_suffix}', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05]); ax.legend(); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f"Final: {bm[-1]:.4f}±{bs[-1]:.4f}",
            transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5), fontsize=10)

    ax = axes[1]
    ax.fill_between(gens, fm - fs2, fm + fs2, alpha=0.2, color='steelblue')
    ax.plot(gens, fm, 's-', color='steelblue', linewidth=2, markersize=6, label='Promedio F1 mediano')
    ax.set_xlabel('Generación'); ax.set_ylabel('F1 Score')
    ax.set_title('Evolución F1 Mediano Promedio', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.05]); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    finals = [t['best_accuracy_history'][-1] if t['best_accuracy_history'] else 0 for t in traces]
    ax.hist(finals, bins=20, color='darkgreen', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(finals), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(finals):.4f}')
    ax.axvline(np.median(finals), color='orange', linestyle='--', linewidth=2, label=f'Mediana: {np.median(finals):.4f}')
    ax.set_xlabel('Mejor Accuracy Final'); ax.set_ylabel('Frecuencia')
    ax.set_title(f'Distribución Accuracies Finales - {group_label}', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  Guardado: {output_path}")


def cmd_ga_evolution(args):
    """Gráficos de evolución GA+TABU+NSGA2 desde directorios de máscaras .pt."""
    import glob as _glob
    from collections import defaultdict
    masks_dirs = args.masks_dirs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_traces = []
    by_replica = defaultdict(list)
    for i, mdir in enumerate(masks_dirs):
        rname = f"R{i+1}"
        pts = _glob.glob(os.path.join(mdir, "mask_*.pt"))
        for pt in pts:
            try:
                t = _gaevo_load_trace(pt)
                all_traces.append(t)
                by_replica[rname].append(t)
            except Exception as e:
                print(f"  Error {os.path.basename(pt)}: {e}")
        print(f"  {rname}: {len(by_replica[rname])} traces cargados")

    if not all_traces:
        print("No se encontraron traces.")
        return

    for rname, traces in sorted(by_replica.items()):
        for t in traces[:3]:
            arch = str(t['architecture']).replace(', ', '-').replace('[', '').replace(']', '')
            fname = f"evolution_individual_{rname}_arch{arch}_sp{t['sparsity']:.2f}.png"
            _gaevo_plot_single(t, str(output_dir / fname), title_suffix=f"{rname} arch={t['architecture']}")

    groups = defaultdict(list)
    for t in all_traces:
        groups[(str(t['architecture']), t['sparsity'])].append(t)
    for (arch, sp), traces in sorted(groups.items()):
        arch_str = str(arch).replace(', ', '-').replace('[', '').replace(']', '')
        fname = f"evolution_aggregated_arch{arch_str}_sp{sp:.2f}.png"
        _gaevo_plot_aggregated(traces, str(output_dir / fname),
                               title_suffix=f"Arch {arch}, Sp={sp:.2f}",
                               group_label=f"{len(traces)} experimentos")

    print(f"Completado. Gráficos en: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# PARETO — visualizaciones de frente de Pareto (GA+Tabu)
# ─────────────────────────────────────────────────────────────────────────────

def _par_load_results(masks_dir):
    results = []
    for mf in Path(masks_dir).glob("mask_*.pt"):
        try:
            data = torch.load(mf, weights_only=False)
            results.append({
                'architecture': str(data['architecture']),
                'sparsity': data['sparsity'],
                'seed_search': data['seed_search'],
                'acc_explore': data['acc_explore'],
                'acc_test': data['acc_test'],
                'trace': data['trace'],
                'mask_file': str(mf),
            })
        except Exception as e:
            print(f"  Error {mf.name}: {e}")
    return results


def _par_plot_evolution(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for r in results:
        trace = r['trace']
        if 'pareto_front_history' not in trace:
            continue
        ph = trace['pareto_front_history']
        fig, ax = plt.subplots(figsize=(12, 8))
        for gen, accs, sparsities in ph[::5]:
            alpha = 0.3 + 0.7 * (gen / max(len(ph), 1))
            ax.scatter(sparsities, [a * 100 for a in accs], alpha=alpha, s=50,
                       label=f'Gen {gen}' if gen % 20 == 0 else None)
        last_gen, last_accs, last_sps = ph[-1]
        ax.scatter(last_sps, [a * 100 for a in last_accs], s=200, c='red',
                   marker='*', edgecolors='black', linewidths=2, label=f'Final (Gen {last_gen})', zorder=10)
        ax.set_xlabel('Sparsity'); ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Evolución Frente de Pareto\n{r["architecture"]}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10); ax.grid(True, alpha=0.3)
        fname = f"pareto_evolution_{r['architecture'].replace(' ','')}_{r['seed_search']}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight'); plt.close()
        print(f"  Guardado: {fname}")


def _par_plot_convergence(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for r in results:
        trace = r['trace']
        evals = trace.get('evaluations', [])
        best_accs = [a * 100 for a in trace.get('best_accuracy_history', [])]
        median_accs = [a * 100 for a in trace.get('median_accuracy_history', [])]
        if not evals or not best_accs:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(evals, best_accs, 'b-', linewidth=2, label='Mejor', marker='o', markersize=4)
        if median_accs:
            ax.plot(evals, median_accs, 'g--', linewidth=2, label='Mediana', marker='s', markersize=4)
        ax.axhline(r['acc_test'] * 100, color='red', linestyle=':',
                   linewidth=2, label=f'Test: {r["acc_test"]*100:.2f}%')
        ax.set_xlabel('Evaluaciones'); ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Convergencia GA+Tabu\n{r["architecture"]} (seed={r["seed_search"]})',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11); ax.grid(True, alpha=0.3)
        fname = f"convergence_{r['architecture'].replace(' ','')}_{r['seed_search']}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight'); plt.close()
        print(f"  Guardado: {fname}")


def _par_plot_scatter(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    by_arch = {}
    for r in results:
        by_arch.setdefault(r['architecture'], []).append(r)
    for arch, items in by_arch.items():
        fig, ax = plt.subplots(figsize=(10, 7))
        sps = [r['trace']['final_stats']['final_sparsity'] for r in items]
        vals = [r['acc_explore'] * 100 for r in items]
        tests = [r['acc_test'] * 100 for r in items]
        ax.scatter(sps, vals, s=150, alpha=0.7, label='Val', marker='o', edgecolors='black')
        ax.scatter(sps, tests, s=150, alpha=0.7, label='Test', marker='s', edgecolors='black')
        ax.set_xlabel('Sparsity'); ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Accuracy vs Sparsity\n{arch}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11); ax.grid(True, alpha=0.3)
        fname = f"scatter_acc_vs_sparsity_{arch.replace(' ','')}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight'); plt.close()
        print(f"  Guardado: {fname}")


def _par_plot_combined(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    by_arch = {}
    for r in results:
        by_arch.setdefault(r['architecture'], []).append(r)
    for arch, items in by_arch.items():
        all_accs, all_sps = [], []
        for r in items:
            trace = r['trace']
            if 'pareto_front_history' not in trace:
                continue
            _, last_accs, last_sps = trace['pareto_front_history'][-1]
            all_accs.extend([a * 100 for a in last_accs])
            all_sps.extend(last_sps)
        if not all_accs:
            continue
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(all_sps, all_accs, s=100, alpha=0.6, edgecolors='black', linewidth=1)
        pts = list(zip(all_sps, all_accs))
        pareto = []
        for i, (si, ai) in enumerate(pts):
            if not any(sj <= si and aj >= ai and (sj < si or aj > ai) for j, (sj, aj) in enumerate(pts) if i != j):
                pareto.append((si, ai))
        if pareto:
            pareto.sort()
            psp, pacc = zip(*pareto)
            ax.plot(psp, pacc, 'r-', linewidth=3, label='Frente de Pareto', zorder=10)
            ax.scatter(psp, pacc, s=200, c='red', marker='*', edgecolors='black', linewidths=2, zorder=11)
        ax.set_xlabel('Sparsity'); ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Frente de Pareto Combinado\n{arch}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11); ax.grid(True, alpha=0.3)
        fname = f"combined_pareto_front_{arch.replace(' ','')}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight'); plt.close()
        print(f"  Guardado: {fname}")


def cmd_pareto(args):
    """Visualizaciones del frente de Pareto para GA+Tabu."""
    masks_dir = args.masks_dir
    output_dir = args.output_dir
    results = _par_load_results(masks_dir)
    if not results:
        print(f"No se encontraron resultados en {masks_dir}")
        return
    print(f"  Cargados {len(results)} resultados")
    _par_plot_evolution(results, output_dir)
    _par_plot_convergence(results, output_dir)
    _par_plot_scatter(results, output_dir)
    _par_plot_combined(results, output_dir)
    print(f"Completado. Gráficos en: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# SS-EVOLUTION — evolución de accuracy SS+TS (Scatter Search + Tabu Search)
# ─────────────────────────────────────────────────────────────────────────────

def _ssevo_load_trace(pt_path):
    data = torch.load(pt_path, weights_only=False)
    trace = data.get('trace', {})
    dense_evals = trace.get('dense_evaluations', trace.get('evaluations', []))
    dense_best  = trace.get('dense_global_best_history', trace.get('best_accuracy_history', []))
    return {
        'evaluations': list(range(1, len(dense_evals) + 1)),
        'accuracy_history': dense_best,
        'best_accuracy_history': dense_best,
        'dense_evaluations': dense_evals,
        'architecture': data.get('architecture', 'unknown'),
        'sparsity': data.get('sparsity', 0.0),
        'acc_explore': data.get('acc_explore', 0.0),
        'acc_test': data.get('acc_test', 0.0),
    }


def _ssevo_plot_single(trace, output_path, title_suffix=""):
    acc = trace['accuracy_history']
    best = trace['best_accuracy_history']
    evals = list(range(1, len(acc) + 1))
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    ax.plot(evals, acc, 'o-', color='steelblue', alpha=0.6, markersize=4, label='Accuracy (Tabu Search)')
    ax.plot(evals, best, 's-', color='darkgreen', linewidth=2.5, markersize=5, label='Mejor accuracy')
    ax.set_xlabel('N° Llamada Función Objetivo (Tabu Search dentro de SS)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy'); ax.set_title(f'Evolución SS+TS - {title_suffix}', fontsize=15, fontweight='bold')
    ax.set_ylim([0, 1.05]); ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    final = best[-1] if best else 0
    ax.text(0.02, 0.98, f"Mejor Val: {final:.4f}\nTest: {trace['acc_test']:.4f}\nEvals: {len(evals)}",
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  Guardado: {output_path}")


def _ssevo_plot_aggregated(traces, output_path, title_suffix="", group_label=""):
    max_len = max(len(t['accuracy_history']) for t in traces)
    acc_mat  = np.full((len(traces), max_len), np.nan)
    best_mat = np.full((len(traces), max_len), np.nan)
    for i, t in enumerate(traces):
        ah = t['accuracy_history'];        acc_mat[i,  :len(ah)] = ah
        bh = t['best_accuracy_history'];   best_mat[i, :len(bh)] = bh
    am, as_ = np.nanmean(acc_mat, 0), np.nanstd(acc_mat, 0)
    bm, bs  = np.nanmean(best_mat, 0), np.nanstd(best_mat, 0)
    evals = list(range(1, max_len + 1))
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))

    ax = axes[0]
    ax.fill_between(evals, am - as_, am + as_, alpha=0.2, color='steelblue')
    ax.plot(evals, am, 'o-', color='steelblue', linewidth=2, markersize=5, label='Promedio accuracy (TS)')
    ax.fill_between(evals, bm - bs, bm + bs, alpha=0.2, color='darkgreen')
    ax.plot(evals, bm, 's-', color='darkgreen', linewidth=2.5, markersize=6, label='Promedio mejor accuracy')
    ax.set_xlabel('N° Llamada Función Objetivo (Tabu Search dentro de SS)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Evolución Promedio SS+TS - {title_suffix}', fontsize=15, fontweight='bold')
    ax.set_ylim([0, 1.05]); ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f"Final: {bm[-1]:.4f}±{bs[-1]:.4f}\nExperimentos: {len(traces)}",
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5), fontsize=10)

    ax = axes[1]
    finals = [t['best_accuracy_history'][-1] if t['best_accuracy_history'] else 0 for t in traces]
    ax.hist(finals, bins=20, color='darkgreen', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(finals), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(finals):.4f}')
    ax.axvline(np.median(finals), color='orange', linestyle='--', linewidth=2, label=f'Mediana: {np.median(finals):.4f}')
    ax.set_xlabel('Mejor Accuracy Final'); ax.set_ylabel('Frecuencia')
    ax.set_title(f'Distribución Accuracies - {group_label}', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  Guardado: {output_path}")


def cmd_ss_evolution(args):
    """Gráficos de evolución SS+TS desde directorio de máscaras .pt."""
    import glob as _glob
    from collections import defaultdict
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    masks_dir = args.masks_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = _glob.glob(os.path.join(masks_dir, "mask_*.pt"))
    if not pt_files:
        print(f"No se encontraron .pt en {masks_dir}")
        return

    all_traces = []
    for pf in pt_files:
        try:
            all_traces.append(_ssevo_load_trace(pf))
        except Exception as e:
            print(f"  Error {os.path.basename(pf)}: {e}")
    print(f"  Cargados {len(all_traces)} traces")

    for t in all_traces[:3]:
        arch = str(t['architecture']).replace(', ', '-').replace('[', '').replace(']', '')
        fname = f"evolution_individual_arch{arch}_sp{t['sparsity']:.2f}.png"
        _ssevo_plot_single(t, str(output_dir / fname),
                           title_suffix=f"Arch {t['architecture']}, Sp={t['sparsity']:.2f}")

    groups = defaultdict(list)
    for t in all_traces:
        groups[(str(t['architecture']), t['sparsity'])].append(t)
    for (arch, sp), traces in sorted(groups.items()):
        arch_str = str(arch).replace(', ', '-').replace('[', '').replace(']', '')
        fname = f"evolution_aggregated_arch{arch_str}_sp{sp:.2f}.png"
        _ssevo_plot_aggregated(traces, str(output_dir / fname),
                               title_suffix=f"Arch {arch}, Sp={sp:.2f}",
                               group_label=f"{len(traces)} experimentos")

    sp_groups = defaultdict(list)
    for t in all_traces:
        sp_groups[t['sparsity']].append(t)
    for sp, traces in sorted(sp_groups.items()):
        fname = f"evolution_global_sp{sp:.2f}.png"
        _ssevo_plot_aggregated(traces, str(output_dir / fname),
                               title_suffix=f"Todas las Arquitecturas, Sp={sp:.2f}",
                               group_label=f"{len(traces)} experimentos totales")

    print(f"Completado. Gráficos en: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON-TABLE — tabla comparativa entre metaheurísticas
# ─────────────────────────────────────────────────────────────────────────────

def _ctbl_calc_metrics(df, name):
    return {
        'name': name,
        'val_acc_mean': df['acc_explore'].mean(), 'val_acc_std': df['acc_explore'].std(),
        'val_acc_best': df['acc_explore'].max(),  'val_acc_worst': df['acc_explore'].min(),
        'test_acc_mean': df['acc_test'].mean(), 'test_acc_std': df['acc_test'].std(),
        'test_acc_best': df['acc_test'].max(),  'test_acc_worst': df['acc_test'].min(),
        'evaluations_mean': df['evaluations'].mean(), 'evaluations_std': df['evaluations'].std(),
        'time_mean': df['time_seconds'].mean(), 'time_std': df['time_seconds'].std(),
        'memory_mean': df['memory_mb'].mean(), 'memory_std': df['memory_mb'].std(),
        'n_runs': len(df), 'n_archs': df['architecture'].nunique(),
        'n_sparsities': df['sparsity'].nunique(),
    }


def _ctbl_save_csv(metrics_list, output_path):
    rows = ['Validation Accuracy (Media)', 'Validation Accuracy (Std)',
            'Validation Accuracy (Mejor)', 'Validation Accuracy (Peor)',
            'Test Accuracy (Media)', 'Test Accuracy (Std)',
            'Test Accuracy (Mejor)', 'Test Accuracy (Peor)',
            'N° Evaluaciones (Media)', 'N° Evaluaciones (Std)',
            'Tiempo (Media) [s]', 'Tiempo (Std) [s]',
            'Memoria (Media) [MB]', 'Memoria (Std) [MB]',
            'Total Runs', 'N° Arquitecturas', 'N° Sparsities']
    keys = ['val_acc_mean', 'val_acc_std', 'val_acc_best', 'val_acc_worst',
            'test_acc_mean', 'test_acc_std', 'test_acc_best', 'test_acc_worst',
            'evaluations_mean', 'evaluations_std', 'time_mean', 'time_std',
            'memory_mean', 'memory_std', 'n_runs', 'n_archs', 'n_sparsities']
    data = {'Métrica': rows}
    for m in metrics_list:
        data[m['name']] = [f"{m[k]:.4f}" if isinstance(m[k], float) else str(m[k]) for k in keys]
    pd.DataFrame(data).to_csv(output_path, index=False)
    print(f"  Guardado CSV: {output_path}")


def _ctbl_save_latex(metrics_list, output_path):
    names = [m['name'] for m in metrics_list]
    header = ' & '.join([r'\textbf{' + n + '}' for n in names])
    lines = [
        r'\begin{table}[h]\centering',
        r'\caption{Comparación entre Metaheurísticas}',
        r'\label{tab:metaheuristics_comparison}',
        r'\begin{tabular}{l' + 'c' * len(names) + '}',
        r'\hline',
        r'\textbf{Métrica} & ' + header + r' \\',
        r'\hline',
    ]
    row_defs = [
        ('Val Accuracy (Media±Std)', lambda m: f"${m['val_acc_mean']:.4f} \\pm {m['val_acc_std']:.4f}$"),
        ('Val Accuracy (Mejor)', lambda m: f"${m['val_acc_best']:.4f}$"),
        ('Test Accuracy (Media±Std)', lambda m: f"${m['test_acc_mean']:.4f} \\pm {m['test_acc_std']:.4f}$"),
        ('Test Accuracy (Mejor)', lambda m: f"${m['test_acc_best']:.4f}$"),
        ('N° Evaluaciones', lambda m: f"${m['evaluations_mean']:.0f} \\pm {m['evaluations_std']:.0f}$"),
        ('Tiempo [s]', lambda m: f"${m['time_mean']:.2f} \\pm {m['time_std']:.2f}$"),
        ('Memoria [MB]', lambda m: f"${m['memory_mean']:.2f} \\pm {m['memory_std']:.2f}$"),
    ]
    for label, fn in row_defs:
        lines.append(label + ' & ' + ' & '.join(fn(m) for m in metrics_list) + r' \\')
        lines.append(r'\hline')
    lines += [r'\end{tabular}', r'\end{table}']
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Guardado LaTeX: {output_path}")


def cmd_comparison_table(args):
    """Tabla comparativa entre GA+TABU+NSGA2, GLS y SS+TABU."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = []
    if args.ga_csvs:
        dfs = [pd.read_csv(p) for p in args.ga_csvs if os.path.exists(p)]
        if dfs:
            df_ga = pd.concat(dfs, ignore_index=True)
            df_ga.drop_duplicates(subset=['architecture', 'sparsity', 'seed_model', 'seed_search', 'replica_id'],
                                  inplace=True)
            metrics_list.append(_ctbl_calc_metrics(df_ga, "GA+TABU+NSGA2"))
            print(f"  GA+TABU+NSGA2: {len(df_ga)} registros únicos")
    if args.gls_csv and os.path.exists(args.gls_csv):
        metrics_list.append(_ctbl_calc_metrics(pd.read_csv(args.gls_csv), "GLS"))
        print(f"  GLS cargado")
    if args.ss_csv and os.path.exists(args.ss_csv):
        metrics_list.append(_ctbl_calc_metrics(pd.read_csv(args.ss_csv), "SS+TABU"))
        print(f"  SS+TABU cargado")

    if len(metrics_list) < 2:
        print("Se necesitan al menos 2 métodos.")
        return

    _ctbl_save_csv(metrics_list, str(output_dir / 'comparison_table.csv'))
    _ctbl_save_latex(metrics_list, str(output_dir / 'comparison_table.tex'))
    print(f"Completado. Archivos en: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# STATS-GA — tests estadísticos para GA+TABU+NSGA2
# ─────────────────────────────────────────────────────────────────────────────

def _sga_test_normality(data, label=""):
    from scipy import stats as _stats
    w, p = _stats.shapiro(data)
    print(f"  Shapiro-Wilk [{label}]: W={w:.4f}, p={p:.4e} ({'normal' if p>0.05 else 'no normal'})")
    return {'statistic': w, 'p_value': p, 'is_normal': p > 0.05}


def _sga_test_baseline(acc, baseline=1/3):
    from scipy import stats as _stats
    t, p = _stats.ttest_1samp(acc, baseline, alternative='greater')
    d = (acc.mean() - baseline) / acc.std()
    print(f"  T-test vs baseline ({baseline:.1%}): t={t:.4f}, p={p:.4e}, Cohen d={d:.3f}")
    return {'t_statistic': t, 'p_value': p, 'cohen_d': d, 'significant': p < 0.05,
            'mean_accuracy': acc.mean(), 'lift_percent': (acc.mean()-baseline)/baseline*100}


def _sga_test_target(acc, target=0.97, epsilon=0.15):
    from scipy import stats as _stats
    _, p_lo = _stats.ttest_1samp(acc, target - epsilon, alternative='greater')
    _, p_hi = _stats.ttest_1samp(acc, target + epsilon, alternative='less')
    p_tost = max(p_lo, p_hi)
    within = int(np.sum(np.abs(acc - target) <= epsilon))
    print(f"  TOST (target={target:.0%}, ε={epsilon:.0%}): p={p_tost:.4e}, dentro_ε={within}/{len(acc)}")
    return {'p_tost': p_tost, 'is_equivalent': p_tost < 0.05,
            'within_epsilon': within, 'total_subnets': len(acc)}


def _sga_test_existence(acc, threshold=0.80):
    from scipy import stats as _stats
    winners = int(np.sum(acc >= threshold))
    p = _stats.binomtest(winners, len(acc), p=0.0, alternative='greater').pvalue
    print(f"  Winning ticket (≥{threshold:.0%}): {winners}/{len(acc)}, max={acc.max():.4f}")
    return {'exists': acc.max() >= threshold, 'winners_count': winners,
            'prob_winner': winners/len(acc), 'max_accuracy': acc.max()}


def _sga_confidence_interval(data, ci=0.95):
    from scipy import stats as _stats
    mean = data.mean()
    se = _stats.sem(data)
    half = se * _stats.t.ppf((1 + ci) / 2, len(data) - 1)
    return mean, mean - half, mean + half


def _sga_anova(df):
    from scipy import stats as _stats
    archs = df['architecture'].unique()
    groups = [df[df['architecture'] == a]['acc_test'].values for a in archs]
    f, p = _stats.f_oneway(*groups)
    print(f"  ANOVA por arquitectura: F={f:.4f}, p={p:.4e} ({'sig.' if p<0.05 else 'no sig.'})")
    return {'f_statistic': f, 'p_value': p, 'significant': p < 0.05}


def _sga_generate_plots(df, output_dir):
    from scipy import stats as _stats
    test_acc = df['acc_test'].values
    val_acc  = df['acc_explore'].values
    out = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bp = axes[0, 0].boxplot([val_acc, test_acc], labels=['Validation', 'Test'], patch_artist=True)
    for patch, c in zip(bp['boxes'], ['lightblue', 'lightgreen']):
        patch.set_facecolor(c)
    axes[0, 0].axhline(1/3, color='red', linestyle='--', linewidth=2, label='Random (33.3%)')
    axes[0, 0].set_ylabel('Accuracy'); axes[0, 0].set_title('Distribución de Accuracy')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].violinplot([val_acc, test_acc], positions=[0, 1], showmeans=True, showmedians=True)
    axes[0, 1].set_xticks([0, 1]); axes[0, 1].set_xticklabels(['Validation', 'Test'])
    axes[0, 1].axhline(1/3, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_ylabel('Accuracy'); axes[0, 1].set_title('Violin Plot'); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(test_acc, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1, 0].axvline(test_acc.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {test_acc.mean():.2%}')
    axes[1, 0].axvline(1/3, color='red', linestyle='--', linewidth=2, label='Random')
    axes[1, 0].set_xlabel('Test Accuracy'); axes[1, 0].set_title('Histograma')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    _stats.probplot(test_acc, dist='norm', plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot'); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / 'distribution_analysis.png', dpi=300, bbox_inches='tight'); plt.close()

    if 'architecture' in df.columns:
        archs = sorted(df['architecture'].unique())
        by_arch = [df[df['architecture'] == a]['acc_test'].values for a in archs]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        bp2 = axes[0].boxplot(by_arch, labels=archs, patch_artist=True)
        for p2 in bp2['boxes']:
            p2.set_facecolor('lightblue')
        axes[0].axhline(1/3, color='red', linestyle='--', linewidth=2, label='Random')
        axes[0].set_xlabel('Architecture'); axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title('Test Accuracy por Arquitectura'); axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45); axes[0].grid(True, alpha=0.3)
        axes[1].violinplot(by_arch, positions=range(len(archs)), showmeans=True, showmedians=True)
        axes[1].set_xticks(range(len(archs))); axes[1].set_xticklabels(archs, rotation=45)
        axes[1].axhline(1/3, color='red', linestyle='--', linewidth=2)
        axes[1].set_ylabel('Test Accuracy'); axes[1].set_title('Violin por Arquitectura')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / 'analysis_by_architecture.png', dpi=300, bbox_inches='tight'); plt.close()

    m, lo, hi = _sga_confidence_interval(test_acc)
    mv, lov, hiv = _sga_confidence_interval(val_acc)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(0, mv, yerr=[[mv-lov], [hiv-mv]], fmt='o', markersize=10, capsize=10, capthick=2,
                color='blue', label='Val Mean ± 95% CI')
    ax.errorbar(1, m, yerr=[[m-lo], [hi-m]], fmt='o', markersize=10, capsize=10, capthick=2,
                color='green', label='Test Mean ± 95% CI')
    ax.axhline(1/3, color='red', linestyle='--', linewidth=2, label='Random (33.3%)')
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 1); ax.set_ylabel('Accuracy')
    ax.set_title('GA+TABU+NSGA2: Intervalos de Confianza (95%)')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Validation', 'Test']); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / 'confidence_intervals.png', dpi=300, bbox_inches='tight'); plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(val_acc, test_acc, alpha=0.6, s=50)
    z = np.polyfit(val_acc, test_acc, 1); pf = np.poly1d(z)
    ax.plot(val_acc, pf(val_acc), 'r--', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    corr = np.corrcoef(val_acc, test_acc)[0, 1]
    ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes, fontsize=12,
            va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Validation Accuracy'); ax.set_ylabel('Test Accuracy')
    ax.set_title('Validation vs Test Accuracy'); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out / 'validation_vs_test.png', dpi=300, bbox_inches='tight'); plt.close()
    print(f"  Gráficos guardados en {output_dir}")


def cmd_stats_ga(args):
    """Tests estadísticos completos para GA+TABU+NSGA2 desde CSV individuales."""
    csv_paths = args.ga_csvs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs = [pd.read_csv(p) for p in csv_paths if os.path.exists(p)]
    if not dfs:
        print("No se encontraron CSV.")
        return
    df = pd.concat(dfs, ignore_index=True)
    if 'replica_id' in df.columns:
        df.drop_duplicates(subset=['architecture', 'sparsity', 'seed_model', 'seed_search', 'replica_id'],
                           inplace=True)
    print(f"  {len(df)} runs cargados")

    test_acc = df['acc_test'].values
    val_acc  = df['acc_explore'].values
    target   = args.target_acc
    epsilon  = args.epsilon
    threshold = args.threshold

    norm_r   = _sga_test_normality(test_acc, "Test Accuracy")
    base_r   = _sga_test_baseline(test_acc)
    target_r = _sga_test_target(test_acc, target=target, epsilon=epsilon)
    exist_r  = _sga_test_existence(test_acc, threshold=threshold)
    anova_r  = _sga_anova(df)
    m, lo, hi = _sga_confidence_interval(test_acc)
    corr = np.corrcoef(val_acc, test_acc)[0, 1]
    print(f"  Pearson val-test: {corr:.3f}")

    _sga_generate_plots(df, str(output_dir))

    slth_ok = target_r['is_equivalent'] and exist_r['exists'] and base_r['significant']
    report = [
        "=" * 70, "REPORTE ESTADÍSTICO - GA+TABU+NSGA2", "=" * 70,
        f"Total runs: {len(df)}", f"Arquitecturas: {df['architecture'].nunique()}",
        f"Red objetivo: {target:.1%}, ε={epsilon:.0%}, threshold={threshold:.0%}",
        "",
        "TEST VS BASELINE ALEATORIO",
        f"  T={base_r['t_statistic']:.4f}, p={base_r['p_value']:.4e}, Cohen d={base_r['cohen_d']:.3f}",
        f"  {'SIGNIFICATIVO' if base_r['significant'] else 'NO SIGNIFICATIVO'}",
        "",
        "TEST DE APROXIMACIÓN A RED OBJETIVO (TOST)",
        f"  p_TOST={target_r['p_tost']:.4e}, dentro_ε={target_r['within_epsilon']}/{target_r['total_subnets']}",
        f"  {'EQUIVALENTE' if target_r['is_equivalent'] else 'NO EQUIVALENTE'}",
        "",
        "TEST DE EXISTENCIA DE WINNING TICKET",
        f"  Max accuracy={exist_r['max_accuracy']:.4f}, winners={exist_r['winners_count']}/{len(test_acc)}",
        f"  {'EXISTE' if exist_r['exists'] else 'NO EXISTE'}",
        "",
        "ANOVA POR ARQUITECTURA",
        f"  F={anova_r['f_statistic']:.4f}, p={anova_r['p_value']:.4e}",
        f"  {'DIFERENCIAS SIGNIFICATIVAS' if anova_r['significant'] else 'SIN DIFERENCIAS'}",
        "",
        "INTERVALO DE CONFIANZA 95% (Test Accuracy)",
        f"  Media={m:.4f}, IC=[{lo:.4f}, {hi:.4f}]",
        "",
        f"CORRELACIÓN VAL-TEST: {corr:.3f}",
        "",
        "CONCLUSIÓN SLTH:",
        f"  {'VALIDADO' if slth_ok else 'NO VALIDADO'}",
    ]
    rpath = output_dir / 'statistical_report.txt'
    rpath.write_text('\n'.join(report), encoding='utf-8')
    print(f"  Guardado: {rpath}")

    pd.DataFrame({'Métrica': ['Mean', 'Std', 'Min', 'Max', 'Normality_p', 'vs_Random_p', 'Cohen_d', 'Corr_Val_Test'],
                  'Valor': [f"{test_acc.mean():.4f}", f"{test_acc.std():.4f}", f"{test_acc.min():.4f}",
                             f"{test_acc.max():.4f}", f"{norm_r['p_value']:.4e}", f"{base_r['p_value']:.4e}",
                             f"{base_r['cohen_d']:.3f}", f"{corr:.3f}"]}).to_csv(
        output_dir / 'statistical_summary.csv', index=False)
    print(f"Completado. Resultados en: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# STATS-HYPERVOLUME — Kruskal-Wallis + Dunn sobre hipervolumen
# ─────────────────────────────────────────────────────────────────────────────

def _shv_kruskal_wallis(groups, names):
    from scipy.stats import kruskal
    valid = [(g, n) for g, n in zip(groups, names) if len(g) > 0]
    if len(valid) < 2:
        return None
    gs = [g for g, _ in valid]; ns = [n for _, n in valid]
    stat, p = kruskal(*gs)
    desc = []
    for g, n in zip(gs, ns):
        desc.append({'Método': n, 'N': len(g), 'Media': np.mean(g),
                     'Mediana': np.median(g), 'Std': np.std(g)})
    return {'statistic': stat, 'p_value': p, 'is_significant': p < 0.05,
            'descriptive': pd.DataFrame(desc), 'methods': ns, 'groups': gs}


def _shv_dunn_test(groups, names, alpha=0.05):
    from itertools import combinations
    from scipy.stats import rankdata, norm
    all_data = np.concatenate(groups)
    ranks = rankdata(all_data)
    n_total = len(all_data)
    group_ranks = []
    idx = 0
    for g in groups:
        group_ranks.append(ranks[idx:idx+len(g)]); idx += len(g)
    pairs = list(combinations(range(len(groups)), 2))
    corr_alpha = alpha / len(pairs)
    rows = []
    for i, j in pairs:
        ni, nj = len(groups[i]), len(groups[j])
        SE = np.sqrt((n_total * (n_total + 1) / 12.0) * (1.0/ni + 1.0/nj))
        z = (np.mean(group_ranks[i]) - np.mean(group_ranks[j])) / SE
        p = 2 * (1 - norm.cdf(abs(z)))
        med_diff = np.median(groups[i]) - np.median(groups[j])
        sig = p < corr_alpha
        if sig:
            winner = names[i] if med_diff > 0 else names[j]
            sig_str = '***' if p < 0.001 else '**' if p < 0.01 else '*'
        else:
            winner, sig_str = 'Sin diferencia', 'ns'
        rows.append({'Comparación': f"{names[i]} vs {names[j]}", 'Z': z, 'p-value': p,
                     'p-ajustado': corr_alpha, 'Significativo': sig,
                     'Δ Mediana': med_diff, 'Ganador': winner, 'Sig.': sig_str})
    return pd.DataFrame(rows)


def _shv_plot(df_combined, output_dir):
    import seaborn as sns
    os.makedirs(output_dir, exist_ok=True)
    methods = df_combined['method'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    data = [df_combined[df_combined['method'] == m]['hypervolume'].values for m in methods]
    bp = axes[0].boxplot(data, labels=methods, patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(methods))):
        patch.set_facecolor(color)
    axes[0].set_ylabel('Hipervolumen', fontweight='bold'); axes[0].set_xlabel('Metaheurística', fontweight='bold')
    axes[0].set_title('Comparación de Hipervolumen por Método', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    sns.violinplot(data=df_combined, x='method', y='hypervolume', ax=axes[1])
    axes[1].set_ylabel('Hipervolumen', fontweight='bold'); axes[1].set_xlabel('Metaheurística', fontweight='bold')
    axes[1].set_title('Distribución de Hipervolumen (Violin)', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hypervolume_comparison_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close(); print(f"  Guardado: hypervolume_comparison_boxplot.png")

    if 'architecture' in df_combined.columns:
        archs = sorted(df_combined['architecture'].unique())
        fig, axes = plt.subplots(1, len(archs), figsize=(5*len(archs), 5))
        if len(archs) == 1:
            axes = [axes]
        for ax, arch in zip(axes, archs):
            da = df_combined[df_combined['architecture'] == arch]
            ms = da['method'].unique()
            d = [da[da['method'] == m]['hypervolume'].values for m in ms]
            bp2 = ax.boxplot(d, labels=ms, patch_artist=True)
            for p2, c in zip(bp2['boxes'], sns.color_palette("husl", len(ms))):
                p2.set_facecolor(c)
            ax.set_title(f'Arq: {arch}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Hipervolumen'); ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hypervolume_by_architecture.png'), dpi=300, bbox_inches='tight')
        plt.close(); print(f"  Guardado: hypervolume_by_architecture.png")


def _shv_report(kw, dunn, output_dir):
    path = os.path.join(output_dir, 'hypervolume_statistical_report.txt')
    lines = ["=" * 70, "REPORTE ESTADÍSTICO: COMPARACIÓN POR HIPERVOLUMEN", "=" * 70, "",
             "1. ESTADÍSTICOS DESCRIPTIVOS", "-" * 70,
             kw['descriptive'].to_string(index=False), "",
             "2. TEST DE KRUSKAL-WALLIS", "-" * 70,
             f"H = {kw['statistic']:.4f}, p = {kw['p_value']:.6f}",
             f"Significativo (α=0.05): {'SÍ' if kw['is_significant'] else 'NO'}", ""]
    if kw['is_significant'] and dunn is not None and not dunn.empty:
        lines += ["3. TEST POST-HOC DE DUNN (Bonferroni)", "-" * 70,
                  dunn.to_string(index=False)]
    lines += ["=" * 70, "FIN"]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Guardado: {path}")


def cmd_stats_hypervolume(args):
    """Tests estadísticos de hipervolumen: Kruskal-Wallis + Dunn entre métodos."""
    import seaborn  # noqa: F401 (ensure available)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    target_archs = ['[13, 128, 3]', '[13, 256, 128, 3]', '[13, 512, 256, 128, 3]']

    dfs = []
    if args.gls_csv and os.path.exists(args.gls_csv):
        df = pd.read_csv(args.gls_csv)
        if 'hypervolume' in df.columns:
            df['method'] = 'GLS'
            dfs.append(df[df['architecture'].isin(target_archs)].copy())
    if args.ss_csv and os.path.exists(args.ss_csv):
        df = pd.read_csv(args.ss_csv)
        if 'hypervolume' in df.columns:
            df['method'] = 'SS+TABU'
            dfs.append(df[df['architecture'].isin(target_archs)].copy())
    if args.ga_dirs:
        arch_map = {'R1': '[13, 128, 3]', 'R2': '[13, 256, 128, 3]', 'R3': '[13, 512, 256, 128, 3]'}
        for i, d in enumerate(args.ga_dirs):
            csv_path = os.path.join(d, 'runs_individual.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['method'] = 'GA+TABU+NSGA2'
                rname = f"R{i+1}"
                if rname in arch_map:
                    df['architecture'] = arch_map[rname]
                if 'final_hv' in df.columns and 'hypervolume' not in df.columns:
                    df['hypervolume'] = df['final_hv']
                if 'hypervolume' in df.columns:
                    dfs.append(df)

    if len(dfs) < 2:
        print("Se necesitan al menos 2 métodos con datos de hipervolumen.")
        return
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"  {len(df_combined)} experimentos totales")

    methods = df_combined['method'].unique().tolist()
    groups = [df_combined[df_combined['method'] == m]['hypervolume'].values for m in methods]
    kw = _shv_kruskal_wallis(groups, methods)
    if kw is None:
        print("Datos insuficientes."); return
    print(f"  Kruskal-Wallis: H={kw['statistic']:.4f}, p={kw['p_value']:.6f}")
    dunn = None
    if kw['is_significant']:
        dunn = _shv_dunn_test(kw['groups'], kw['methods'])
        print(dunn.to_string(index=False))

    _shv_plot(df_combined, output_dir)
    _shv_report(kw, dunn, output_dir)
    print(f"Completado. Resultados en: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# STATS-COMPARE — análisis estadístico comparativo entre metaheurísticas
# ─────────────────────────────────────────────────────────────────────────────

def _sc_test_normality(data, name):
    from scipy.stats import shapiro
    w, p = shapiro(data)
    return {'method': name, 'statistic': w, 'p_value': p,
            'is_normal': p > 0.05, 'interpretation': 'Normal' if p > 0.05 else 'No normal'}


def _sc_test_levene(*groups):
    from scipy.stats import levene
    stat, p = levene(*groups)
    return {'statistic': stat, 'p_value': p, 'is_homogeneous': p > 0.05,
            'interpretation': 'Varianzas homogéneas' if p > 0.05 else 'Varianzas heterogéneas'}


def _sc_kruskal_wallis(*groups, group_names):
    from scipy.stats import kruskal
    stat, p = kruskal(*groups)
    desc = [{'group': n, 'n': len(g), 'mean': np.mean(g), 'median': np.median(g), 'std': np.std(g)}
            for g, n in zip(groups, group_names)]
    return {'statistic': stat, 'p_value': p, 'is_significant': p < 0.05,
            'descriptive': pd.DataFrame(desc)}


def _sc_mann_whitney(groups, names):
    from itertools import combinations
    from scipy.stats import mannwhitneyu, norm
    results = []
    pairs = list(combinations(range(len(groups)), 2))
    corr_alpha = 0.05 / len(pairs)
    for i, j in pairs:
        stat, p = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
        n1, n2 = len(groups[i]), len(groups[j])
        z = norm.ppf(1 - p/2)
        eff = abs(z) / np.sqrt(n1 + n2)
        sig = p < corr_alpha
        results.append({'comparison': f"{names[i]} vs {names[j]}", 'U_statistic': stat,
                        'p_value': p, 'is_significant': sig,
                        'median_diff': np.median(groups[i]) - np.median(groups[j]),
                        'effect_size_r': eff})
    return pd.DataFrame(results)


def _sc_cohen_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    s = np.sqrt(((n1-1)*np.var(g1, ddof=1) + (n2-1)*np.var(g2, ddof=1)) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / s


def _sc_generate_plots(df, output_dir):
    import seaborn as sns
    os.makedirs(output_dir, exist_ok=True)
    methods_order = [m for m in ['GLS', 'GA+TABU+NSGA2', 'SS+TABU'] if m in df['method'].values]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.boxplot(data=df, x='method', y='acc_test_mean', order=methods_order, ax=axes[0,0], palette='Set2')
    axes[0,0].set_title('Test Accuracy por Metaheurística', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Metaheurística'); axes[0,0].set_ylabel('Test Accuracy')
    axes[0,0].grid(True, alpha=0.3, axis='y')
    sns.boxplot(data=df, x='method', y='acc_explore_mean', order=methods_order, ax=axes[0,1], palette='Set2')
    axes[0,1].set_title('Validation Accuracy por Metaheurística', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Metaheurística'); axes[0,1].set_ylabel('Validation Accuracy')
    axes[0,1].grid(True, alpha=0.3, axis='y')
    if 'architecture' in df.columns:
        sns.boxplot(data=df, x='architecture', y='acc_test_mean', hue='method', ax=axes[1,0], palette='Set2')
        axes[1,0].set_title('Test Accuracy por Arquitectura y Método', fontsize=14, fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45); axes[1,0].grid(True, alpha=0.3, axis='y')
    sns.violinplot(data=df, x='method', y='acc_test_mean', order=methods_order, ax=axes[1,1], palette='Set2')
    axes[1,1].set_title('Violin Plot - Test Accuracy', fontsize=14, fontweight='bold')
    axes[1,1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close(); print(f"  Guardado: comparison_distributions.png")

    if 'architecture' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        for m in methods_order:
            mdata = df[df['method'] == m].groupby('architecture')['acc_test_mean'].mean()
            ax.plot(range(len(mdata)), mdata.values, marker='o', linewidth=2, markersize=8, label=m)
        ax.set_xticks(range(len(mdata)))
        ax.set_xticklabels(mdata.index, rotation=45, ha='right')
        ax.set_title('Interacción: Método × Arquitectura', fontsize=14, fontweight='bold')
        ax.set_xlabel('Arquitectura'); ax.set_ylabel('Test Accuracy Promedio')
        ax.legend(title='Método'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'interaction_method_architecture.png'), dpi=300, bbox_inches='tight')
        plt.close(); print(f"  Guardado: interaction_method_architecture.png")


def _sc_generate_report(df, output_dir, posthoc_df=None):
    os.makedirs(output_dir, exist_ok=True)
    lines = ["=" * 70, "REPORTE ESTADÍSTICO COMPARATIVO: Metaheurísticas", "=" * 70]
    for method in df['method'].unique():
        md = df[df['method'] == method]
        lines += [f"\n{method}:",
                  f"  Experimentos: {len(md)}",
                  f"  Test Accuracy: {md['acc_test_mean'].mean():.4f} ± {md['acc_test_mean'].std():.4f}",
                  f"  Val Accuracy:  {md['acc_explore_mean'].mean():.4f} ± {md['acc_explore_mean'].std():.4f}",
                  f"  Rango test:    [{md['acc_test_mean'].min():.4f}, {md['acc_test_mean'].max():.4f}]"]
    ranking = df.groupby('method')['acc_test_mean'].mean().sort_values(ascending=False)
    lines += ["\nRANKING (por accuracy media):"]
    for i, (m, v) in enumerate(ranking.items(), 1):
        lines.append(f"  {i}. {m}: {v:.4f}")
    if posthoc_df is not None and not posthoc_df.empty:
        lines += ["\nCOMPARACIONES POST-HOC (Mann-Whitney U):", posthoc_df.to_string(index=False)]
    path = os.path.join(output_dir, 'statistical_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Guardado: {path}")


def cmd_stats_compare(args):
    """Tests estadísticos comparativos entre GA+TABU+NSGA2, GLS y SS+TABU."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    target_archs = ['[13, 128, 3]', '[13, 256, 128, 3]', '[13, 512, 256, 128, 3]']
    arch_to_rep = {'[13, 128, 3]': 'R1', '[13, 256, 128, 3]': 'R2', '[13, 512, 256, 128, 3]': 'R3'}
    arch_map_ga = {'R1': '[13, 128, 3]', 'R2': '[13, 256, 128, 3]', 'R3': '[13, 512, 256, 128, 3]'}

    all_dfs = []
    if args.ga_dirs:
        for i, d in enumerate(args.ga_dirs):
            csv_path = os.path.join(d, 'runs_aggregated.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['method'] = 'GA+TABU+NSGA2'
                rname = f"R{i+1}"
                if rname in arch_map_ga:
                    df['architecture'] = arch_map_ga[rname]
                df['replica'] = rname
                all_dfs.append(df)
                print(f"  GA {rname}: {len(df)} experimentos")
    if args.gls_agg and os.path.exists(args.gls_agg):
        df = pd.read_csv(args.gls_agg)
        df['method'] = 'GLS'
        df = df[df['architecture'].isin(target_archs)].copy()
        df['replica'] = df['architecture'].map(arch_to_rep)
        all_dfs.append(df); print(f"  GLS: {len(df)} experimentos")
    if args.ss_agg and os.path.exists(args.ss_agg):
        df = pd.read_csv(args.ss_agg)
        df['method'] = 'SS+TABU'
        df = df[df['architecture'].isin(target_archs)].copy()
        df['replica'] = df['architecture'].map(arch_to_rep)
        all_dfs.append(df); print(f"  SS+TABU: {len(df)} experimentos")

    if not all_dfs:
        print("No se encontraron datos."); return
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total: {len(df_combined)} experimentos, métodos: {df_combined['method'].unique().tolist()}")

    methods = df_combined['method'].unique().tolist()
    groups = [df_combined[df_combined['method'] == m]['acc_test_mean'].values for m in methods]

    kw = _sc_kruskal_wallis(*groups, group_names=methods)
    print(f"  Kruskal-Wallis: H={kw['statistic']:.4f}, p={kw['p_value']:.6e}, "
          f"{'SIGNIFICATIVO' if kw['is_significant'] else 'NO SIGNIFICATIVO'}")

    posthoc = None
    if kw['is_significant']:
        posthoc = _sc_mann_whitney(groups, methods)
        posthoc.to_csv(os.path.join(output_dir, 'posthoc_comparisons.csv'), index=False)
        print(f"  Post-hoc guardado: posthoc_comparisons.csv")

    _sc_generate_plots(df_combined, output_dir)
    _sc_generate_report(df_combined, output_dir, posthoc)
    df_combined.to_csv(os.path.join(output_dir, 'combined_data.csv'), index=False)
    print(f"Completado. Resultados en: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="generate_diagrams",
        description="Genera todos los diagramas y análisis del proyecto SLTH."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # convergence
    c = sub.add_parser("convergence", help="Gráficos de convergencia desde .pt")
    c.add_argument("--results-dir", default="results", help="Directorio raíz de resultados")
    c.add_argument("--method", default=None, help="Filtrar por método (ej. GLS)")
    c.add_argument("--top", type=int, default=None, help="Solo top-N resultados")
    c.add_argument("--output-dir", default="results/plots/convergence")

    # evolution
    e = sub.add_parser("evolution", help="Evolución promedio / por sparsity / top-bottom")
    e.add_argument("--masks-dir", required=True, help="Directorio con .pt (ej. results/GLS/best_masks)")
    e.add_argument("--output-dir", default="results/plots/evolution")

    # replicas
    r = sub.add_parser("replicas", help="Convergencia + boxplot por configuración de réplicas")
    r.add_argument("individual_csv", help="CSV con resultados individuales")
    r.add_argument("aggregated_csv", help="CSV con resultados agregados")
    r.add_argument("--output-dir", default=None)

    # comparison
    cp = sub.add_parser("comparison", help="4 figuras comparativas entre dos métodos")
    cp.add_argument("--gls-csv", required=True, help="CSV del método base")
    cp.add_argument("--opt-csv", required=True, help="CSV del método optimizado")
    cp.add_argument("--output-dir", default="results/plots/comparison")

    # stats
    st = sub.add_parser("stats", help="Tests estadísticos SLTH")
    st.add_argument("--results-dir", default="results/GLS_optimized")
    st.add_argument("--target-acc", type=float, default=0.95)
    st.add_argument("--epsilon", type=float, default=0.35)
    st.add_argument("--threshold", type=float, default=0.60)

    # export-csv
    ex = sub.add_parser("export-csv", help="Exporta trazas .pt a CSV")
    ex.add_argument("--results-dir", default="results")
    ex.add_argument("--method", default=None)
    ex.add_argument("--top", type=int, default=None)
    ex.add_argument("--output-dir", default="results/convergence_histories")

    # sparsity
    sp = sub.add_parser("sparsity", help="Accuracy vs sparsity y análisis de variabilidad")
    sp.add_argument("aggregated_csv", help="CSV de resultados agregados")
    sp.add_argument("--individual-csv", default=None, help="CSV individual (opcional)")
    sp.add_argument("--output-dir", default="results/plots/sparsity")

    # diversity
    dv = sub.add_parser("diversity", help="Análisis de diversidad Hamming entre máscaras")
    dv.add_argument("--masks-dirs", nargs="+",
                    default=["results/GLS-Correciones/best_masks", "results/SS-Optuna-Final/best_masks"])
    dv.add_argument("--output-dir", default="results/plots/diversity")
    dv.add_argument("--max-samples", type=int, default=100)

    # neurons
    neu = sub.add_parser("neurons", help="Análisis de neuronas activas/inactivas por capa")
    neu.add_argument("--masks-dirs", nargs="+",
                     default=[f"results/GA_TABU_NSGA2-R{i}/best_masks" for i in range(1, 5)])
    neu.add_argument("--output-dir", default="results/plots/neurons")

    # boxplots
    bp = sub.add_parser("boxplots", help="Boxplots unificados GA+TABU+NSGA2 por arquitectura")
    bp.add_argument("--ga-csvs", nargs="+",
                    default=[f"results/GA_TABU_NSGA2-OPTUNA-R{i}/runs_individual.csv" for i in range(1, 5)])
    bp.add_argument("--output-dir", default="results/plots/boxplots")

    # ga-evolution
    ge = sub.add_parser("ga-evolution", help="Evolución GA+TABU+NSGA2 desde máscaras .pt")
    ge.add_argument("--masks-dirs", nargs="+",
                    default=[f"results/GA_TABU_NSGA2-OPTUNA-R{i}/best_masks" for i in range(1, 5)])
    ge.add_argument("--output-dir", default="results/plots/ga_evolution")

    # pareto
    pt = sub.add_parser("pareto", help="Visualizaciones del frente de Pareto (GA+Tabu)")
    pt.add_argument("--masks-dir", default="results/GA_TABU/best_masks")
    pt.add_argument("--output-dir", default="results/plots/pareto")

    # ss-evolution
    ss = sub.add_parser("ss-evolution", help="Evolución SS+TS desde máscaras .pt")
    ss.add_argument("--masks-dir", default="results/SS-1/best_masks")
    ss.add_argument("--output-dir", default="results/plots/ss_evolution")

    # comparison-table
    ct = sub.add_parser("comparison-table", help="Tabla comparativa entre metaheurísticas (CSV + LaTeX)")
    ct.add_argument("--ga-csvs", nargs="+",
                    default=[f"results/GA_TABU_NSGA2-OPTUNA-R{i}/runs_individual.csv" for i in range(1, 5)])
    ct.add_argument("--gls-csv", default=None)
    ct.add_argument("--ss-csv", default=None)
    ct.add_argument("--output-dir", default="results/metaheuristics_comparison")

    # stats-ga
    sg = sub.add_parser("stats-ga", help="Tests estadísticos para GA+TABU+NSGA2")
    sg.add_argument("--ga-csvs", nargs="+",
                    default=[f"results/GA_TABU_NSGA2-OPTUNA-R{i}/runs_individual.csv" for i in range(1, 4)])
    sg.add_argument("--output-dir", default="results/stats_ga")
    sg.add_argument("--target-acc", type=float, default=0.97)
    sg.add_argument("--epsilon", type=float, default=0.15)
    sg.add_argument("--threshold", type=float, default=0.80)

    # stats-hypervolume
    shv = sub.add_parser("stats-hypervolume", help="Kruskal-Wallis + Dunn sobre hipervolumen")
    shv.add_argument("--gls-csv", default=None)
    shv.add_argument("--ss-csv", default=None)
    shv.add_argument("--ga-dirs", nargs="+",
                     default=[f"results/GA_TABU_NSGA2-OPTUNA-R{i}" for i in range(1, 4)])
    shv.add_argument("--output-dir", default="results/stats_hypervolume")

    # stats-compare
    sc = sub.add_parser("stats-compare", help="Tests estadísticos comparativos entre metaheurísticas")
    sc.add_argument("--ga-dirs", nargs="+",
                    default=[f"results/GA_TABU_NSGA2-OPTUNA-R{i}" for i in range(1, 4)])
    sc.add_argument("--gls-agg", default=None, help="CSV agregado GLS")
    sc.add_argument("--ss-agg", default=None, help="CSV agregado SS+TABU")
    sc.add_argument("--output-dir", default="results/stats_compare")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Ruta default para output de replicas
    if args.cmd == "replicas" and args.output_dir is None:
        args.output_dir = str(Path(args.individual_csv).parent / "plots")

    dispatch = {
        "convergence":      cmd_convergence,
        "evolution":        cmd_evolution,
        "replicas":         cmd_replicas,
        "comparison":       cmd_comparison,
        "stats":            cmd_stats,
        "export-csv":       cmd_export_csv,
        "sparsity":         cmd_sparsity,
        "diversity":        cmd_diversity,
        "neurons":          cmd_neurons,
        "boxplots":         cmd_boxplots,
        "ga-evolution":     cmd_ga_evolution,
        "pareto":           cmd_pareto,
        "ss-evolution":     cmd_ss_evolution,
        "comparison-table": cmd_comparison_table,
        "stats-ga":         cmd_stats_ga,
        "stats-hypervolume": cmd_stats_hypervolume,
        "stats-compare":    cmd_stats_compare,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
