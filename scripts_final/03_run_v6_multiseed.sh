#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 03_run_v6_multiseed.sh
# Corre V6 parametrizado en varias seeds, secuencialmente
# ============================================================

source scripts_final/00_config_server.sh

MULTI_OUT_ROOT="${OUT_ROOT}/v6_multiseed"
mkdir -p "${MULTI_OUT_ROOT}" "${LOG_DIR}"

echo "========================================"
echo "V6 MULTISEED"
echo "========================================"
echo "Seeds: ${V6_SEEDS}"
echo "OUT_ROOT=${MULTI_OUT_ROOT}"
echo "CORES=${CORES}"
echo "========================================"

for SEED in ${V6_SEEDS}; do

  OUT_DIR="${MULTI_OUT_ROOT}/seed_${SEED}"
  LOG_FILE="${LOG_DIR}/v6_multiseed_seed_${SEED}.log"

  echo ""
  echo "----------------------------------------"
  echo "Iniciando seed ${SEED}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "LOG_FILE=${LOG_FILE}"
  echo "Inicio: $(date)"
  echo "----------------------------------------"

  mkdir -p "${OUT_DIR}"

  Rscript "${TRAIN_SCRIPT}" \
    --seed "${SEED}" \
    --data_dir "${DATA_DIR}" \
    --out_dir "${OUT_DIR}" \
    --rf_random "${MULTI_RF_RANDOM}" \
    --rf_trees "${MULTI_RF_TREES}" \
    --xgb_random "${MULTI_XGB_RANDOM}" \
    --xgb_refine "${MULTI_XGB_REFINE}" \
    --xgb_nrounds_1 "${MULTI_XGB_NROUNDS_1}" \
    --xgb_nrounds_2 "${MULTI_XGB_NROUNDS_2}" \
    --xgb_early_1 "${MULTI_XGB_EARLY_1}" \
    --xgb_early_2 "${MULTI_XGB_EARLY_2}" \
    --cores "${CORES}" \
    > "${LOG_FILE}" 2>&1

  echo "Seed ${SEED} terminada: $(date)"
  echo "Métricas seed ${SEED}:"
  cat "${OUT_DIR}/model_comparison_v6.csv"

done

echo ""
echo "========================================"
echo "MULTISEED COMPLETADO"
echo "========================================"
echo "Resultados en: ${MULTI_OUT_ROOT}"