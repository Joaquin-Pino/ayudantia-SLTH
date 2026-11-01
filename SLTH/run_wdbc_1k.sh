#!/usr/bin/env bash
# Corre GLS, SS+TS y NSGA-II+TS (1k evaluaciones) sobre WDBC, en secuencia.
# Uso (desde cualquier carpeta): bash SLTH/run_wdbc_1k.sh   |   log: SLTH/log_wdbc_1k.txt
cd "$(dirname "$0")" || exit 1
PY=/home/joaquin/Documents/u/ayudantia_investigacion/.venv/bin/python

for c in 1-experiment_gls_wdbc 2-experiment_ss_wdbc 3-experiment_ga_tabu_nsga2_wdbc; do
  "$PY" -u src/run_experiment.py "configs/1k/wdbc/$c.yaml" || echo "FALLO: $c"
done
