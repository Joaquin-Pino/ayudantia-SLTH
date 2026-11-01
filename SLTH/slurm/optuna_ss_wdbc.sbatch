#!/usr/bin/env bash
# Búsqueda de hiperparámetros con Optuna sobre WDBC solo para SS (con refset_quality_ratio = 0.69).
# Un único estudio de Optuna en serie. Sobrescribe results/wdbc/optuna_ss/.
#
# Por qué se repite: la primera corrida (26-10-2026, 2 h 46 min) usó refset_quality_ratio = 0.5, pero
# la Tabla 2 del paper lo fija en 0.69. Se corrigió en tune_optuna.py y GLS y GA no se repiten.
# Los resultados con 0.5 no sirven; cópialos antes si quieres compararlos (results/wdbc/optuna_ss_ref05).
# El estudio vive en memoria (sin storage): si el job muere por tiempo, se pierden los 51 trials.
#
# Uso (desde SLTH/):
#   mkdir -p logs && sbatch slurm/optuna_ss_wdbc.sbatch
#   PY=/ruta/a/python sbatch slurm/optuna_ss_wdbc.sbatch   # otro intérprete
#
# Resultados en results/wdbc/optuna_ss/  |  logs en logs/optuna_ss_wdbc_<job>.out
#
#SBATCH --job-name=optuna_ss_wdbc
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=GPU

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Si el cluster usa módulos, cargarlos aquí (p. ej. `module load python/3.11`)
PY=${PY:-../.venv/bin/python}

# torch usa por defecto todos los núcleos del nodo; limitarlo a los CPUs asignados
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Método: ss | nodo: $(hostname) | inicio: $(date)"
"$PY" -u src/tune_optuna.py ss \
  --config configs/optuna_ss_wdbc.yaml \
  --output-dir results/wdbc/optuna_ss
echo "Fin: $(date)"
