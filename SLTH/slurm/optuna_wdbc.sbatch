#!/usr/bin/env bash
# Búsqueda de hiperparámetros con Optuna sobre WDBC: un job array con una tarea por método
# (0 = GLS, 1 = SS, 2 = GA+TABU+NSGA2). Cada tarea corre un estudio de Optuna en serie.
#
# Los tiempos por método difieren (GLS ~1 h, SS y GA ~8 h): lánzalo con `bash slurm/lanzar_wdbc.sh optuna`.
#
# Uso (desde SLTH/):
#   mkdir -p logs && sbatch slurm/optuna_wdbc.sbatch
#   sbatch --array=1 slurm/optuna_wdbc.sbatch          # solo SS
#   PY=/ruta/a/python sbatch slurm/optuna_wdbc.sbatch  # otro intérprete
#
# Resultados en results/wdbc/optuna_<metodo>/  |  logs en logs/optuna_wdbc_<job>_<tarea>.out
#
#SBATCH --job-name=optuna_wdbc
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
#SBATCH --partition=GPU

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Si el cluster usa módulos, cargarlos aquí (p. ej. `module load python/3.11`)
PY=${PY:-../.venv/bin/python}

METHODS=(gls ss ga_tabu_nsga2)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

# torch usa por defecto todos los núcleos del nodo; limitarlo a los CPUs asignados
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Método: $METHOD | nodo: $(hostname) | inicio: $(date)"
"$PY" -u src/tune_optuna.py "$METHOD" \
  --config "configs/optuna_${METHOD}_wdbc.yaml" \
  --output-dir "results/wdbc/optuna_${METHOD}"
echo "Fin: $(date)"
