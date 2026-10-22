#!/usr/bin/env bash
# Corre GLS, SS+TS y NSGA-II+TS (1k evaluaciones) sobre BCW, en secuencia.
# Uso (desde cualquier carpeta): bash SLTH/run_bcw_1k.sh   |   log: SLTH/log_bcw_1k.txt
cd "$(dirname "$0")" || exit 1
PY=/home/joaquin/Documents/u/ayudantia_investigacion/.venv/bin/python

for c in 1-experiment_gls_bcw 2-experiment_ss_bcw 3-experiment_ga_tabu_nsga2_bcw; do
  "$PY" -u src/run_experiment.py "configs/1k/bcw/$c.yaml" || echo "FALLO: $c"
done
