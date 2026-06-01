suppressPackageStartupMessages({
  library(data.table)
})

cat("\n========================================\n")
cat("CHECK MULTISEED V6\n")
cat("========================================\n")

base_dir <- "resultados_server/v6_multiseed"
out_dir  <- "resultados_server/v6_multiseed_consolidated"

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

required_files <- c(
  "model_comparison_v6.csv",
  "fixtures_2026_predictions_all_models_v6.csv",
  "test_predictions_all_models_v6.csv",
  "confusion_matrices_all_models_v6.csv",
  "run_config_v6.csv"
)

seed_dirs <- list.dirs(base_dir, recursive = FALSE, full.names = TRUE)
seed_dirs <- seed_dirs[grepl("seed_[0-9]+$", seed_dirs)]

if (length(seed_dirs) == 0) {
  stop("No se encontraron carpetas seed_* en: ", base_dir)
}

seed_info <- data.table(
  seed_dir = seed_dirs,
  seed = as.integer(sub(".*seed_", "", seed_dirs))
)[order(seed)]

cat("\nSeeds detectadas:\n")
print(seed_info)

# ============================================================
# 1. Verificación de archivos por seed
# ============================================================

check_list <- lapply(seq_len(nrow(seed_info)), function(i) {
  sd <- seed_info$seed_dir[i]
  seed <- seed_info$seed[i]
  
  data.table(
    seed = seed,
    file = required_files,
    exists = file.exists(file.path(sd, required_files))
  )
})

check_dt <- rbindlist(check_list)

fwrite(
  check_dt,
  file.path(out_dir, "multiseed_file_check_v6.csv")
)

cat("\nVerificación de archivos:\n")
print(dcast(check_dt, seed ~ file, value.var = "exists"))

missing_dt <- check_dt[exists == FALSE]

if (nrow(missing_dt) > 0) {
  cat("\nATENCIÓN: faltan archivos:\n")
  print(missing_dt)
  stop("Hay seeds incompletas. Revisar multiseed_file_check_v6.csv")
}

# ============================================================
# 2. Consolidar métricas por modelo y seed
# ============================================================

metrics_all <- rbindlist(lapply(seq_len(nrow(seed_info)), function(i) {
  f <- file.path(seed_info$seed_dir[i], "model_comparison_v6.csv")
  dt <- fread(f)
  dt[, seed := seed_info$seed[i]]
  dt
}), fill = TRUE)

setcolorder(metrics_all, c("seed", setdiff(names(metrics_all), "seed")))

fwrite(
  metrics_all,
  file.path(out_dir, "multiseed_model_metrics_all_v6.csv")
)

metric_cols <- intersect(
  c("Accuracy", "BalancedAccuracy", "LogLoss"),
  names(metrics_all)
)

if (!"model" %in% names(metrics_all)) {
  stop("No existe columna 'model' en model_comparison_v6.csv")
}

metrics_summary <- metrics_all[
  ,
  c(
    .N,
    lapply(.SD, mean, na.rm = TRUE),
    lapply(.SD, sd, na.rm = TRUE)
  ),
  by = model,
  .SDcols = metric_cols
]

# Renombrar columnas
old_names <- names(metrics_summary)
mean_names <- paste0("mean_", metric_cols)
sd_names   <- paste0("sd_", metric_cols)

setnames(
  metrics_summary,
  old = c(metric_cols, paste0(metric_cols, ".1")),
  new = c(mean_names, sd_names)
)

# Ranking recomendado para MC:
# menor LogLoss, luego mayor BalancedAccuracy, luego mayor Accuracy
if ("mean_LogLoss" %in% names(metrics_summary)) {
  setorder(metrics_summary, mean_LogLoss, -mean_BalancedAccuracy, -mean_Accuracy)
} else {
  setorder(metrics_summary, -mean_BalancedAccuracy, -mean_Accuracy)
}

metrics_summary[, rank_for_mc := .I]

fwrite(
  metrics_summary,
  file.path(out_dir, "multiseed_model_metrics_summary_v6.csv")
)

cat("\nResumen multiseed por modelo:\n")
print(metrics_summary)

best_model <- metrics_summary[rank_for_mc == 1, model]

cat("\nModelo candidato para MC:\n")
cat(best_model, "\n")

writeLines(
  best_model,
  file.path(out_dir, "best_model_for_mc_v6.txt")
)

# ============================================================
# 3. Consolidar predicciones fixtures 2026
# ============================================================

fixtures_all <- rbindlist(lapply(seq_len(nrow(seed_info)), function(i) {
  f <- file.path(seed_info$seed_dir[i], "fixtures_2026_predictions_all_models_v6.csv")
  dt <- fread(f)
  dt[, seed := seed_info$seed[i]]
  dt
}), fill = TRUE)

setcolorder(fixtures_all, c("seed", setdiff(names(fixtures_all), "seed")))

fwrite(
  fixtures_all,
  file.path(out_dir, "multiseed_fixtures_predictions_all_v6.csv")
)

cat("\nColumnas en fixtures predictions:\n")
print(names(fixtures_all))

# ============================================================
# 4. Detectar columnas de probabilidad
# ============================================================

prob_cols <- grep(
  "prob|p_home|p_draw|p_away|HomeWin|Draw|AwayWin",
  names(fixtures_all),
  value = TRUE,
  ignore.case = TRUE
)

prob_cols <- setdiff(prob_cols, c("predicted_class", "pred_class", "prediction"))

cat("\nColumnas candidatas de probabilidad:\n")
print(prob_cols)

# Columnas de identificación posibles
id_candidates <- c(
  "match_id", "fixture_id", "game_id",
  "group", "round", "stage",
  "home_team", "away_team",
  "team_home", "team_away",
  "home", "away",
  "date"
)

id_cols <- intersect(id_candidates, names(fixtures_all))

if (length(id_cols) == 0) {
  stop("No se detectaron columnas ID de partido. Revisar nombres de fixtures_2026_predictions_all_models_v6.csv")
}

# Si hay columna model, promediar solo el mejor modelo
if ("model" %in% names(fixtures_all)) {
  fixtures_model <- fixtures_all[model == best_model]
  
  if (nrow(fixtures_model) == 0) {
    cat("\nNo encontré exactamente el best_model en fixtures. Modelos disponibles:\n")
    print(unique(fixtures_all$model))
    stop("El nombre del modelo en fixtures no coincide con model_comparison.")
  }
  
} else {
  fixtures_model <- copy(fixtures_all)
}

# Mantener solo columnas numéricas candidatas
prob_cols_num <- prob_cols[sapply(fixtures_model[, ..prob_cols], is.numeric)]

if (length(prob_cols_num) == 0) {
  stop("No se detectaron columnas numéricas de probabilidad.")
}

cat("\nColumnas numéricas usadas para promedio:\n")
print(prob_cols_num)

# ============================================================
# 5. Promedio y SD por partido entre seeds
# ============================================================

mean_dt <- fixtures_model[
  ,
  lapply(.SD, mean, na.rm = TRUE),
  by = id_cols,
  .SDcols = prob_cols_num
]

sd_dt <- fixtures_model[
  ,
  lapply(.SD, sd, na.rm = TRUE),
  by = id_cols,
  .SDcols = prob_cols_num
]

setnames(
  mean_dt,
  old = prob_cols_num,
  new = paste0(prob_cols_num, "_mean")
)

setnames(
  sd_dt,
  old = prob_cols_num,
  new = paste0(prob_cols_num, "_sd")
)

fixtures_mean <- merge(mean_dt, sd_dt, by = id_cols, all = TRUE)

# ============================================================
# 6. Intentar crear predicción final por máxima probabilidad
# ============================================================

mean_prob_cols <- grep("_mean$", names(fixtures_mean), value = TRUE)

home_col <- mean_prob_cols[grepl("home|HomeWin", mean_prob_cols, ignore.case = TRUE)][1]
draw_col <- mean_prob_cols[grepl("draw|Draw", mean_prob_cols, ignore.case = TRUE)][1]
away_col <- mean_prob_cols[grepl("away|AwayWin", mean_prob_cols, ignore.case = TRUE)][1]

if (!is.na(home_col) && !is.na(draw_col) && !is.na(away_col)) {
  fixtures_mean[
    ,
    pred_class_mean := c("HomeWin", "Draw", "AwayWin")[
      max.col(.SD, ties.method = "first")
    ],
    .SDcols = c(home_col, draw_col, away_col)
  ]
  
  fixtures_mean[
    ,
    prob_max_mean := do.call(pmax, .SD),
    .SDcols = c(home_col, draw_col, away_col)
  ]
  
  fixtures_mean[
    ,
    prob_margin_mean := {
      mat <- as.matrix(.SD)
      sorted <- t(apply(mat, 1, sort, decreasing = TRUE))
      sorted[, 1] - sorted[, 2]
    },
    .SDcols = c(home_col, draw_col, away_col)
  ]
}

fwrite(
  fixtures_mean,
  file.path(out_dir, "multiseed_fixtures_probabilities_v6.csv")
)

cat("\nArchivo principal generado:\n")
cat(file.path(out_dir, "multiseed_fixtures_probabilities_v6.csv"), "\n")

# ============================================================
# 7. Partidos inestables
# ============================================================

sd_prob_cols <- grep("_sd$", names(fixtures_mean), value = TRUE)

if (length(sd_prob_cols) > 0) {
  fixtures_mean[
    ,
    max_prob_sd := do.call(pmax, .SD),
    .SDcols = sd_prob_cols
  ]
  
  unstable <- fixtures_mean[
    order(-max_prob_sd)
  ][1:min(.N, 30)]
  
  fwrite(
    unstable,
    file.path(out_dir, "multiseed_unstable_matches_v6.csv")
  )
  
  cat("\nTop partidos más inestables:\n")
  print(unstable)
}

# ============================================================
# 8. Chequeo de suma de probabilidades
# ============================================================

if (!is.na(home_col) && !is.na(draw_col) && !is.na(away_col)) {
  fixtures_mean[
    ,
    prob_sum := get(home_col) + get(draw_col) + get(away_col)
  ]
  
  prob_check <- fixtures_mean[
    ,
    .(
      min_prob_sum = min(prob_sum, na.rm = TRUE),
      mean_prob_sum = mean(prob_sum, na.rm = TRUE),
      max_prob_sum = max(prob_sum, na.rm = TRUE)
    )
  ]
  
  fwrite(
    prob_check,
    file.path(out_dir, "multiseed_probability_sum_check_v6.csv")
  )
  
  cat("\nChequeo suma de probabilidades:\n")
  print(prob_check)
}

cat("\n========================================\n")
cat("CHECK MULTISEED V6 COMPLETADO\n")
cat("Salidas en: ", out_dir, "\n")
cat("========================================\n")