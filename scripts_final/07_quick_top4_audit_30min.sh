#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 07_quick_top4_audit_30min.sh
# Auditoría rápida de estabilidad Monte Carlo para top 4
# ============================================================

SEEDS=(3001 3002 3003 3004 3005)
NSIM=10000

BASE_DIR="resultados_server/v6_multiseed_consolidated"
OFFICIAL_TAG="MONTE_CARLO_V6_MULTISEED_OFFICIAL_ANNEXC"
KO_CACHE_NAME="mc_KO_dynamic_prob_precomputed_all_pairs_v6MS.csv"
OFFICIAL_KO_CACHE="${BASE_DIR}/${OFFICIAL_TAG}/${KO_CACHE_NAME}"

MC_SCRIPT="scripts_final/05_mc_worldcup_v6_multiseed_annexC.R"

mkdir -p logs

TOTAL=${#SEEDS[@]}
START_ALL=$(date +%s)

progress_bar() {
  local current=$1
  local total=$2
  local width=30
  local percent=$(( current * 100 / total ))
  local filled=$(( current * width / total ))
  local empty=$(( width - filled ))

  printf "["
  if [[ "$filled" -gt 0 ]]; then
    printf "%0.s#" $(seq 1 "$filled")
  fi
  if [[ "$empty" -gt 0 ]]; then
    printf "%0.s-" $(seq 1 "$empty")
  fi
  printf "] %3d%% (%d/%d)" "$percent" "$current" "$total"
}

elapsed_fmt() {
  local elapsed=$1
  printf "%02d:%02d:%02d" \
    $((elapsed/3600)) \
    $(((elapsed%3600)/60)) \
    $((elapsed%60))
}

echo "======================================"
echo "AUDITORÍA RÁPIDA TOP 4"
echo "Seeds: ${SEEDS[*]}"
echo "N_SIM=${NSIM}"
echo "Total corridas: ${TOTAL}"
echo "BASE_DIR=${BASE_DIR}"
echo "MC_SCRIPT=${MC_SCRIPT}"
echo "KO cache oficial=${OFFICIAL_KO_CACHE}"
echo "======================================"
echo ""

if [[ ! -f "${MC_SCRIPT}" ]]; then
  echo "ERROR: no existe el script MC:"
  echo "${MC_SCRIPT}"
  exit 1
fi

if [[ -f "${OFFICIAL_KO_CACHE}" ]]; then
  echo "OK: se encontró cache KO oficial."
  ls -lh "${OFFICIAL_KO_CACHE}"
else
  echo "AVISO: no se encontró cache KO oficial."
  echo "El script R recalculará KO en cada seed."
  echo "Ruta esperada:"
  echo "${OFFICIAL_KO_CACHE}"
fi

echo ""

# ============================================================
# Evitar procesos duplicados accidentales
# ============================================================

RUNNING_R=$(pgrep -f "05_mc_worldcup_v6_multiseed_annexC.R" || true)
RUNNING_SH=$(pgrep -f "07_quick_top4_audit_30min.sh" || true)

if [[ -n "${RUNNING_R}" ]]; then
  echo "AVISO: hay procesos R del Monte Carlo corriendo:"
  ps -f -p ${RUNNING_R} || true
  echo ""
  echo "Mátalos antes de relanzar si son corridas viejas:"
  echo "pkill -f 05_mc_worldcup_v6_multiseed_annexC.R"
  echo ""
  exit 1
fi

# ============================================================
# Correr seeds secuenciales
# ============================================================

i=0

for S in "${SEEDS[@]}"; do
  i=$((i + 1))

  TAG="QUICK_TOP4_AUDIT_SEED_${S}"
  LOG="logs/quick_top4_seed_${S}.log"

  CURRENT_OUT_DIR="${BASE_DIR}/${TAG}"
  CURRENT_KO_CACHE="${CURRENT_OUT_DIR}/${KO_CACHE_NAME}"

  echo "--------------------------------------"
  echo "Seed ${i}/${TOTAL}: MC_SEED=${S}"
  echo "TAG=${TAG}"
  echo "LOG=${LOG}"
  echo "OUT_DIR=${CURRENT_OUT_DIR}"
  echo "Progreso global:"
  progress_bar "$((i - 1))" "$TOTAL"
  echo ""
  echo "--------------------------------------"

  mkdir -p "${CURRENT_OUT_DIR}"

  if [[ -f "${OFFICIAL_KO_CACHE}" ]]; then
    echo "Copiando cache KO oficial a carpeta de esta seed..."
    cp -f "${OFFICIAL_KO_CACHE}" "${CURRENT_KO_CACHE}"
  else
    echo "Sin cache KO oficial disponible. El R recalculará KO."
  fi

  if [[ -f "${CURRENT_KO_CACHE}" ]]; then
    echo "OK: cache KO disponible para esta seed:"
    ls -lh "${CURRENT_KO_CACHE}"
  else
    echo "AVISO: no hay cache KO en carpeta de seed."
  fi

  echo ""

  START_SEED=$(date +%s)

  MC_SEED="${S}" \
  N_SIM="${NSIM}" \
  MC_OUT_TAG="${TAG}" \
  Rscript "${MC_SCRIPT}" \
    > "${LOG}" 2>&1 &

  PID=$!

  spin='-\|/'
  k=0

  while kill -0 "$PID" 2>/dev/null; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_SEED))
    TOTAL_ELAPSED=$((NOW - START_ALL))

    printf "\rSeed %s corriendo %s | seed elapsed %s | total elapsed %s" \
      "$S" \
      "${spin:k++%${#spin}:1}" \
      "$(elapsed_fmt "$ELAPSED")" \
      "$(elapsed_fmt "$TOTAL_ELAPSED")"

    sleep 2
  done

  wait "$PID"
  EXIT_CODE=$?

  NOW=$(date +%s)
  ELAPSED=$((NOW - START_SEED))

  if [[ "$EXIT_CODE" -eq 0 ]]; then
    printf "\rSeed %s terminada OK | tiempo %s%60s\n" \
      "$S" "$(elapsed_fmt "$ELAPSED")" ""
  else
    printf "\rSeed %s falló | tiempo %s%60s\n" \
      "$S" "$(elapsed_fmt "$ELAPSED")" ""

    echo ""
    echo "Revisa el log:"
    echo "${LOG}"
    echo ""
    echo "Últimas líneas:"
    tail -n 80 "${LOG}" || true
    exit "$EXIT_CODE"
  fi

  echo ""
  echo "Chequeo rápido del log KO/simulación:"
  grep -E "CARGANDO PROBABILIDADES KO|PRECÁLCULO DE PROBABILIDADES KO|Cruces KO cargados|Cruces ordenados|Simulación|simulación|Listo|ERROR|Error" "${LOG}" || true

  echo ""
  echo "Progreso global:"
  progress_bar "$i" "$TOTAL"
  echo ""
  echo ""
done

# ============================================================
# Resumen final
# ============================================================

echo "======================================"
echo "Resumiendo resultados"
echo "======================================"

Rscript - <<'RSCRIPT'
suppressPackageStartupMessages({
  library(data.table)
})

base_dir <- "resultados_server/v6_multiseed_consolidated"
seeds <- c(3001, 3002, 3003, 3004, 3005)

cat("Leyendo resultados de seeds:", paste(seeds, collapse = ", "), "\n")

find_file <- function(seed, pattern) {
  d <- file.path(base_dir, paste0("QUICK_TOP4_AUDIT_SEED_", seed))
  f <- list.files(d, pattern = pattern, full.names = TRUE)
  if (length(f) == 0) return(NA_character_)
  f[1]
}

pick_col <- function(dt, patterns) {
  nms <- names(dt)
  low <- tolower(nms)

  for (p in patterns) {
    hit <- grep(p, low)
    if (length(hit) > 0) return(nms[hit[1]])
  }

  NA_character_
}

read_stage <- function(seed) {
  cat("  - Leyendo seed", seed, "\n")

  f <- find_file(seed, "mc_team_stage_probabilities.*\\.csv$")

  if (is.na(f)) {
    stop("No encuentro stage probabilities para seed ", seed)
  }

  dt <- fread(f)

  team_col <- pick_col(
    dt,
    c("^team$", "selection", "seleccion", "squad", "country")
  )

  if (is.na(team_col)) {
    stop("No encuentro columna de equipo en ", f)
  }

  champ_col <- pick_col(dt, c("champion", "campeon", "winner"))
  final_col <- pick_col(dt, c("final"))
  sf_col    <- pick_col(dt, c("semi", "sf"))
  qf_col    <- pick_col(dt, c("quarter", "qf", "cuarto"))

  out <- data.table(
    seed = seed,
    team = dt[[team_col]]
  )

  if (!is.na(champ_col)) out[, p_champion := as.numeric(dt[[champ_col]])]
  if (!is.na(final_col)) out[, p_final := as.numeric(dt[[final_col]])]
  if (!is.na(sf_col))    out[, p_sf := as.numeric(dt[[sf_col]])]
  if (!is.na(qf_col))    out[, p_qf := as.numeric(dt[[qf_col]])]

  out
}

all_stage <- rbindlist(
  lapply(seeds, read_stage),
  fill = TRUE
)

prob_cols <- grep("^p_", names(all_stage), value = TRUE)

for (cc in prob_cols) {
  if (max(all_stage[[cc]], na.rm = TRUE) > 1.5) {
    all_stage[, (cc) := get(cc) / 100]
  }
}

if ("p_champion" %in% names(all_stage)) {
  all_stage[, rank_champion := frank(-p_champion, ties.method = "min"), by = seed]
}

if ("p_final" %in% names(all_stage)) {
  all_stage[, rank_final := frank(-p_final, ties.method = "min"), by = seed]
}

if ("p_sf" %in% names(all_stage)) {
  all_stage[, rank_sf := frank(-p_sf, ties.method = "min"), by = seed]
}

summary_team <- all_stage[
  ,
  .(
    mean_champion = if ("p_champion" %in% names(.SD)) mean(p_champion, na.rm = TRUE) else NA_real_,
    sd_champion   = if ("p_champion" %in% names(.SD)) sd(p_champion, na.rm = TRUE) else NA_real_,
    min_champion  = if ("p_champion" %in% names(.SD)) min(p_champion, na.rm = TRUE) else NA_real_,
    max_champion  = if ("p_champion" %in% names(.SD)) max(p_champion, na.rm = TRUE) else NA_real_,

    mean_final    = if ("p_final" %in% names(.SD)) mean(p_final, na.rm = TRUE) else NA_real_,
    sd_final      = if ("p_final" %in% names(.SD)) sd(p_final, na.rm = TRUE) else NA_real_,
    mean_sf       = if ("p_sf" %in% names(.SD)) mean(p_sf, na.rm = TRUE) else NA_real_,
    sd_sf         = if ("p_sf" %in% names(.SD)) sd(p_sf, na.rm = TRUE) else NA_real_,

    min_rank_champion = if ("rank_champion" %in% names(.SD)) min(rank_champion, na.rm = TRUE) else NA_real_,
    max_rank_champion = if ("rank_champion" %in% names(.SD)) max(rank_champion, na.rm = TRUE) else NA_real_,
    min_rank_sf       = if ("rank_sf" %in% names(.SD)) min(rank_sf, na.rm = TRUE) else NA_real_,
    max_rank_sf       = if ("rank_sf" %in% names(.SD)) max(rank_sf, na.rm = TRUE) else NA_real_
  ),
  by = team
][order(-mean_champion)]

out_dir <- file.path(base_dir, "QUICK_TOP4_AUDIT_SUMMARY")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

fwrite(
  all_stage,
  file.path(out_dir, "quick_top4_all_stage_by_seed.csv")
)

fwrite(
  summary_team,
  file.path(out_dir, "quick_top4_summary_by_team.csv")
)

cat("\n==============================\n")
cat("TOP 10 por probabilidad media de campeón\n")
cat("==============================\n")
print(summary_team[1:min(10, .N)])

if ("mean_sf" %in% names(summary_team)) {
  cat("\n==============================\n")
  cat("TOP 10 por probabilidad media de semifinal / top 4\n")
  cat("==============================\n")
  print(summary_team[order(-mean_sf)][1:min(10, .N)])
}

target <- c("Argentina", "Spain", "France", "England", "Portugal", "Brazil", "Colombia", "Belgium")
target_dt <- summary_team[team %in% target]

cat("\n==============================\n")
cat("Bloque de apuesta / favoritos ampliado\n")
cat("==============================\n")
print(target_dt[order(-mean_champion)])

cat("\n==============================\n")
cat("Ranking de campeón por seed, top 8\n")
cat("==============================\n")

if ("rank_champion" %in% names(all_stage)) {
  print(
    all_stage[
      rank_champion <= 8,
      .(seed, rank_champion, team, p_champion)
    ][order(seed, rank_champion)]
  )
} else {
  cat("No existe rank_champion porque no se detectó columna de campeón.\n")
}

cat("\nArchivos guardados en:\n")
cat(out_dir, "\n")
RSCRIPT

END_ALL=$(date +%s)
TOTAL_TIME=$((END_ALL - START_ALL))

echo ""
echo "======================================"
echo "Listo."
echo "Tiempo total: $(elapsed_fmt "$TOTAL_TIME")"
echo "======================================"