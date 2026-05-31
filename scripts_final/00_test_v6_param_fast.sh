#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 00_test_v6_param_fast.sh
# Prueba rápida del V6 parametrizado
# ============================================================

source scripts_final/00_config_server.sh

SEED=123
OUT_DIR="${OUT_ROOT}/test_seed_${SEED}_fast"
LOG_FILE="${LOG_DIR}/test_seed_${SEED}_fast.log"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "========================================"
echo "TEST RÁPIDO V6 PARAMETRIZADO"
echo "========================================"
echo "SEED=${SEED}"
echo "OUT_DIR=${OUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "========================================"

Rscript "${TRAIN_SCRIPT}" \
  --seed "${SEED}" \
  --data_dir "${DATA_DIR}" \
  --out_dir "${OUT_DIR}" \
  --rf_random "${FAST_RF_RANDOM}" \
  --rf_trees "${FAST_RF_TREES}" \
  --xgb_random "${FAST_XGB_RANDOM}" \
  --xgb_refine "${FAST_XGB_REFINE}" \
  --xgb_nrounds_1 "${FAST_XGB_NROUNDS_1}" \
  --xgb_nrounds_2 "${FAST_XGB_NROUNDS_2}" \
  --xgb_early_1 "${FAST_XGB_EARLY_1}" \
  --xgb_early_2 "${FAST_XGB_EARLY_2}" \
  --cores "${CORES}" \
  > "${LOG_FILE}" 2>&1

echo ""
echo "========================================"
echo "TEST TERMINADO"
echo "========================================"

echo ""
echo "Archivos generados:"
ls -lh "${OUT_DIR}"

echo ""
echo "Métricas:"
cat "${OUT_DIR}/model_comparison_v6.csv"

echo ""
echo "Últimas 60 líneas del log:"
tail -n 60 "${LOG_FILE}"