#!/usr/bin/env bash

# ============================================================
# 00_config_server.sh
# Configuración global para correr V6 en servidor
# ============================================================

# Raíz del proyecto.
# Si ejecutas desde la raíz del repo, esto funciona.
export PROJECT_DIR="$(pwd)"

# Carpetas principales
export DATA_DIR="${PROJECT_DIR}/datos"
export SCRIPTS_DIR="${PROJECT_DIR}/scripts_final"
export OUT_ROOT="${PROJECT_DIR}/resultados_server"
export LOG_DIR="${PROJECT_DIR}/logs"

# Script R parametrizado
export TRAIN_SCRIPT="${SCRIPTS_DIR}/01_train_v6_single_seed_param.R"

# Recursos
export CORES=3

# BLAS / OpenMP
export OMP_NUM_THREADS="${CORES}"
export OPENBLAS_NUM_THREADS="${CORES}"
export MKL_NUM_THREADS="${CORES}"
export VECLIB_MAXIMUM_THREADS="${CORES}"
export NUMEXPR_NUM_THREADS="${CORES}"

# Parámetros V6: modo prueba rápida
export FAST_RF_RANDOM=2
export FAST_RF_TREES=100
export FAST_XGB_RANDOM=2
export FAST_XGB_REFINE=1
export FAST_XGB_NROUNDS_1=100
export FAST_XGB_NROUNDS_2=150
export FAST_XGB_EARLY_1=20
export FAST_XGB_EARLY_2=20

# Parámetros V6: corrida completa single seed
export FULL_RF_RANDOM=25
export FULL_RF_TREES=900
export FULL_XGB_RANDOM=30
export FULL_XGB_REFINE=15
export FULL_XGB_NROUNDS_1=1200
export FULL_XGB_NROUNDS_2=1500
export FULL_XGB_EARLY_1=50
export FULL_XGB_EARLY_2=60

# Parámetros recomendados para multiseed futuro
export MULTI_RF_RANDOM=15
export MULTI_RF_TREES=700
export MULTI_XGB_RANDOM=20
export MULTI_XGB_REFINE=8
export MULTI_XGB_NROUNDS_1=1000
export MULTI_XGB_NROUNDS_2=1200
export MULTI_XGB_EARLY_1=50
export MULTI_XGB_EARLY_2=60

# Seeds futuras para refinamiento
export V6_SEEDS="101 123 202 303 404 505 606 707 808 909"

# Crear carpetas si no existen
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

echo "========================================"
echo "CONFIG SERVIDOR CARGADA"
echo "========================================"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "SCRIPTS_DIR=${SCRIPTS_DIR}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "CORES=${CORES}"
echo "========================================"
