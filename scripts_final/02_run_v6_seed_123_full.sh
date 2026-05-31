#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 02_run_v6_seed_123_full.sh
# Corrida completa V6 seed 123
# ============================================================

source scripts_final/00_config_server.sh

SEED=123
OUT_DIR="${OUT_ROOT}/v6_seed_${SEED}"
LOG_FILE="${LOG_DIR}/v6_seed_${SEED}.log"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "========================================"
echo "V6 FULL SINGLE SEED"
echo "========================================"
echo "SEED=${SEED}"
echo "OUT_DIR=${OUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "CORES=${CORES}"
echo "========================================"

Rscript "${TRAIN_SCRIPT}" \
  --seed "${SEED}" \
  --data_dir "${DATA_DIR}" \
  --out_dir "${OUT_DIR}" \
  --rf_random "${FULL_RF_RANDOM}" \
  --rf_trees "${FULL_RF_TREES}" \
  --xgb_random "${FULL_XGB_RANDOM}" \
  --xgb_refine "${FULL_XGB_REFINE}" \
  --xgb_nrounds_1 "${FULL_XGB_NROUNDS_1}" \
  --xgb_nrounds_2 "${FULL_XGB_NROUNDS_2}" \
  --xgb_early_1 "${FULL_XGB_EARLY_1}" \
  --xgb_early_2 "${FULL_XGB_EARLY_2}" \
  --cores "${CORES}" \
  > "${LOG_FILE}" 2>&1

echo ""
echo "========================================"
echo "V6 FULL TERMINADO"
echo "========================================"

echo ""
echo "Métricas:"
cat "${OUT_DIR}/model_comparison_v6.csv"

echo ""
echo "Últimas 80 líneas del log:"
tail -n 80 "${LOG_FILE}"