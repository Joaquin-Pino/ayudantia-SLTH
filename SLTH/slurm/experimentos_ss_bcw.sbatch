#!/usr/bin/env bash
# Experimentos de BCW con 10k y 50k evaluaciones para Scatter Search (SS): un job array con una
# tarea por config (0 = SS 10k, 1 = SS 50k). Cada tarea corre run_experiment.py con su config;
# las rutas de salida están en el `output` del YAML.
#
# Uso (desde SLTH/):
#   mkdir -p logs && sbatch slurm/experimentos_ss_bcw.sbatch
#   sbatch --array=0 slurm/experimentos_ss_bcw.sbatch   # solo el de 10k
#   PY=/ruta/a/python sbatch slurm/experimentos_ss_bcw.sbatch
#
# Resultados en results/bcw/{10k,50k}/2-SS-Base/...  |  logs en logs/experimentos_ss_bcw_<job>_<tarea>.out
#
#SBATCH --job-name=experimentos_ss_bcw
#SBATCH --array=0-1
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
  configs/10k/bcw/2-experiment_ss_bcw.yaml
  configs/50k/bcw/2-experiment_ss_bcw.yaml
)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

# torch usa por defecto todos los núcleos del nodo; limitarlo a los CPUs asignados
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Config: $CONFIG | nodo: $(hostname) | inicio: $(date)"
"$PY" -u src/run_experiment.py "$CONFIG"
echo "Fin: $(date)"
