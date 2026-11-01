#!/usr/bin/env bash
# Lanza todos los trabajos de WDBC pidiendo a cada uno su propio tiempo (ver estimaciones abajo).
# Uso (desde SLTH/):   bash slurm/lanzar_wdbc.sh [todo|exp|optuna]      (por defecto: todo)
#
# IMPORTANTE: los experimentos usan los hiperparámetros escritos en los YAML de configs/. Si
# relanzas Optuna, copia sus mejores parámetros (results/wdbc/optuna_*/best_*_params.yaml) a esos YAML
# ANTES de lanzar los experimentos; este script no lo hace ni encadena las dependencias.
#
# Estimaciones (tiempo de las corridas previas en results/wdbc, suma de time_seconds, con margen ~2x):
#   GLS 10k  1.6 h -> 4 h      GLS 50k  1.8 h -> 4 h
#   GA  10k  2.1 h -> 5 h      GA  50k  3.9 h (3 arquitecturas, como el paper) -> 8 h
#   SS  10k  ~14 h (sin corridas previas: extrapolado desde Optuna) -> 24 h
#   SS  50k  ~28 h (3 arq.), extrapolado -> 48 h
#   Optuna: GLS 0.2 h -> 1 h | SS 3.0 h -> 8 h | GA 2.8 h -> 8 h  (sin storage: si muere por tiempo se pierde todo)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
WHAT=${1:-todo}

if [[ $WHAT == todo || $WHAT == optuna ]]; then
  sbatch --array=0 --time=01:00:00 slurm/optuna_wdbc.sbatch   # GLS
  sbatch --array=1 --time=08:00:00 slurm/optuna_wdbc.sbatch   # SS
  sbatch --array=2 --time=08:00:00 slurm/optuna_wdbc.sbatch   # GA+TABU+NSGA2
fi

if [[ $WHAT == todo || $WHAT == exp ]]; then
  sbatch --array=0 --time=04:00:00 slurm/experimentos_wdbc.sbatch   # GLS 10k
  sbatch --array=1 --time=24:00:00 slurm/experimentos_wdbc.sbatch   # SS  10k
  sbatch --array=2 --time=05:00:00 slurm/experimentos_wdbc.sbatch   # GA  10k
  sbatch --array=3 --time=04:00:00 slurm/experimentos_wdbc.sbatch   # GLS 50k
  sbatch --array=4 --time=48:00:00 slurm/experimentos_wdbc.sbatch   # SS  50k
  sbatch --array=5 --time=08:00:00 slurm/experimentos_wdbc.sbatch   # GA  50k
fi
