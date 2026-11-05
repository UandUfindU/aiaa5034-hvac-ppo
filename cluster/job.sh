#!/bin/bash
#SBATCH --partition=acd_u
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -euo pipefail
cd /data/user/yfeng044/projects/aiaa5034-hvac-ppo
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
PY=/data/user/yfeng044/projects/aiaa5034-hvac-assets/venv/bin/python
ROOT=/data/user/yfeng044/projects/aiaa5034-hvac-artifacts
mode=$1
case "$mode" in
 preflight)
  "$PY" -m pytest -q
  "$PY" -m src.experiments --smoke --device cpu --output-dir "$ROOT/smoke_cpu"
  "$PY" -m src.experiments --smoke --device cuda --output-dir "$ROOT/smoke_cuda"
  "$PY" -m src.aggregate --input-dir "$ROOT/smoke_cpu" --output-dir "$ROOT/smoke_cpu/tables"
  "$PY" -m src.plot --input-dir "$ROOT/smoke_cpu/tables" --output-dir "$ROOT/smoke_cpu/figures"
  ;;
 full)
  seed=$2
  "$PY" -m src.experiments --full --device cpu --seed "$seed" --total-timesteps 100000 --output-dir "$ROOT/full"
  ;;
 aggregate)
  "$PY" -m src.aggregate --input-dir "$ROOT/full" --output-dir "$ROOT/full/tables"
  "$PY" -m src.plot --input-dir "$ROOT/full/tables" --output-dir "$ROOT/full/figures"
  ;;
 *) exit 2;;
esac
