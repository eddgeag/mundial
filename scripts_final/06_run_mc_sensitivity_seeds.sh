#!/usr/bin/env bash
set -euo pipefail

SEEDS=(1001 1002 1003 1004 1005 1006 1007 1008 1009 1010)
NSIM=100000

mkdir -p logs

for S in "${SEEDS[@]}"; do
  TAG="MONTE_CARLO_SENSITIVITY_SEED_${S}"

  echo "======================================"
  echo "Corriendo sensibilidad MC_SEED=${S}"
  echo "N_SIM=${NSIM}"
  echo "TAG=${TAG}"
  echo "======================================"

  MC_SEED="${S}" \
  N_SIM="${NSIM}" \
  MC_OUT_TAG="${TAG}" \
  Rscript scripts_final/05_mc_worldcup_v6_multiseed_annexC.R \
    > "logs/mc_sensitivity_seed_${S}.log" 2>&1
done