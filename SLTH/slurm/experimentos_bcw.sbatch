#!/usr/bin/env bash
# Experimentos de BCW (dedup: true, split 75/25) para GLS, SS y GA+Tabu+NSGA-II con 10k y 50k
# evaluaciones: un job array con una tarea por config:
#   0 = GLS 10k | 1 = SS 10k | 2 = GA 10k | 3 = GLS 50k | 4 = SS 50k | 5 = GA 50k
# Cada tarea corre run_experiment.py con su config; las rutas de salida están en el `output` del YAML.
#
# Los tiempos son MUY distintos entre tareas, y un array solo admite un --time en la cabecera, así que
# lánzalo con el script que pide el tiempo adecuado a cada tarea:
#   bash slurm/lanzar_bcw.sh              # experimentos + optuna
#   bash slurm/lanzar_bcw.sh exp          # solo experimentos
# o a mano (desde SLTH/):
#   mkdir -p logs && sbatch --array=0 --time=04:00:00 slurm/experimentos_bcw.sbatch
#   PY=/ruta/a/python sbatch ...
#
# Resultados en results/bcw/{10k,50k}/...  |  logs en logs/experimentos_bcw_<job>_<tarea>.out
#
#SBATCH --job-name=experimentos_bcw
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
#SBATCH --partition=GPU

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Si el cluster usa módulos, cargarlos aquí (p. ej. `module load python/3.11`)
PY=${PY:-../.venv/bin/python}

CONFIGS=(
  configs/10k/bcw/1-experiment_gls_bcw.yaml
  configs/10k/bcw/2-experiment_ss_bcw.yaml
  configs/10k/bcw/3-experiment_ga_tabu_nsga2_bcw.yaml
  configs/50k/bcw/1-experiment_gls_bcw.yaml
  configs/50k/bcw/2-experiment_ss_bcw.yaml
  configs/50k/bcw/3-experiment_ga_tabu_nsga2_bcw.yaml
)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

# torch usa por defecto todos los núcleos del nodo; limitarlo a los CPUs asignados
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Config: $CONFIG | nodo: $(hostname) | inicio: $(date)"
"$PY" -u src/run_experiment.py "$CONFIG"
echo "Fin: $(date)"
