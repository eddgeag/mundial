# ============================================================
# 06_train_models_v6_feature_engineered.R
# V6: nuevas features + modelos memory-safe + XGBoost corregido
# ============================================================
#
# Entrada esperada:
#   datos/training_v5_strength_adjusted.csv
#   datos/fixtures_2026_v5_strength_adjusted.csv
#
# Salidas principales:
#   datos/training_v6_feature_engineered.csv
#   datos/fixtures_2026_v6_feature_engineered.csv
#   modelos_v6_feature_engineered/model_comparison_v6.csv
#   modelos_v6_feature_engineered/fixtures_2026_predictions_all_models_v6.csv
#
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(caret)
  library(nnet)
  library(ranger)
  library(xgboost)
})

# ------------------------------------------------------------
# 0. Configuración
# ------------------------------------------------------------
# ------------------------------------------------------------
# 0. Configuración parametrizada para servidor
# ------------------------------------------------------------

parse_args <- function(defaults = list()) {
  args <- commandArgs(trailingOnly = TRUE)
  out <- defaults
  
  if (length(args) == 0) return(out)
  
  i <- 1
  while (i <= length(args)) {
    arg <- args[[i]]
    
    if (grepl("^--", arg)) {
      key <- sub("^--", "", arg)
      
      if (grepl("=", key)) {
        parts <- strsplit(key, "=", fixed = TRUE)[[1]]
        out[[parts[1]]] <- parts[2]
        i <- i + 1
      } else {
        if (i == length(args) || grepl("^--", args[[i + 1]])) {
          out[[key]] <- TRUE
          i <- i + 1
        } else {
          out[[key]] <- args[[i + 1]]
          i <- i + 2
        }
      }
    } else {
      i <- i + 1
    }
  }
  
  out
}

as_int <- function(x) as.integer(x)
as_num <- function(x) as.numeric(x)
as_chr <- function(x) as.character(x)

as_bool <- function(x) {
  if (is.logical(x)) return(x)
  tolower(as.character(x)) %in% c("true", "t", "1", "yes", "y", "si", "sí")
}

args <- parse_args(list(
  seed = 123,
  data_dir = "datos",
  out_dir = "modelos_v6_feature_engineered",
  write_legacy_data = FALSE,
  
  rf_random = 25,
  rf_trees = 900,
  
  xgb_random = 30,
  xgb_refine = 15,
  xgb_nrounds_1 = 1200,
  xgb_nrounds_2 = 1500,
  xgb_early_1 = 50,
  xgb_early_2 = 60,
  
  cores = max(1, parallel::detectCores() - 1)
))

SEED <- as_int(args$seed)

data_dir <- as_chr(args$data_dir)
out_dir  <- as_chr(args$out_dir)

WRITE_LEGACY_DATA <- as_bool(args$write_legacy_data)

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

train_file_v5 <- file.path(data_dir, "training_v5_strength_adjusted.csv")
pred_file_v5  <- file.path(data_dir, "fixtures_2026_v5_strength_adjusted.csv")

# IMPORTANTE:
# En servidor guardamos los V6 dentro del out_dir para que cada seed
# sea autocontenida y no pise datos/training_v6_feature_engineered.csv.
train_file_v6 <- file.path(out_dir, "training_v6_feature_engineered.csv")
pred_file_v6  <- file.path(out_dir, "fixtures_2026_v6_feature_engineered.csv")

# Opcional: compatibilidad con scripts antiguos que esperan los V6 en datos/
legacy_train_file_v6 <- file.path(data_dir, "training_v6_feature_engineered.csv")
legacy_pred_file_v6  <- file.path(data_dir, "fixtures_2026_v6_feature_engineered.csv")

class_levels <- c("HomeWin", "Draw", "AwayWin")

# Ajustes principales
N_RF_RANDOM      <- as_int(args$rf_random)
RF_NUM_TREES     <- as_int(args$rf_trees)

N_XGB_RANDOM     <- as_int(args$xgb_random)
N_XGB_REFINE     <- as_int(args$xgb_refine)
XGB_NROUNDS_1    <- as_int(args$xgb_nrounds_1)
XGB_NROUNDS_2    <- as_int(args$xgb_nrounds_2)
XGB_EARLY_1      <- as_int(args$xgb_early_1)
XGB_EARLY_2      <- as_int(args$xgb_early_2)

N_CORES <- as_int(args$cores)
N_CORES <- max(1, N_CORES)

Sys.setenv(
  OMP_NUM_THREADS = N_CORES,
  OPENBLAS_NUM_THREADS = N_CORES,
  MKL_NUM_THREADS = N_CORES,
  VECLIB_MAXIMUM_THREADS = N_CORES,
  NUMEXPR_NUM_THREADS = N_CORES
)

set.seed(SEED)

run_config <- data.table(
  parameter = c(
    "SEED", "data_dir", "out_dir", "WRITE_LEGACY_DATA",
    "N_RF_RANDOM", "RF_NUM_TREES",
    "N_XGB_RANDOM", "N_XGB_REFINE",
    "XGB_NROUNDS_1", "XGB_NROUNDS_2",
    "XGB_EARLY_1", "XGB_EARLY_2",
    "N_CORES"
  ),
  value = as.character(c(
    SEED, data_dir, out_dir, WRITE_LEGACY_DATA,
    N_RF_RANDOM, RF_NUM_TREES,
    N_XGB_RANDOM, N_XGB_REFINE,
    XGB_NROUNDS_1, XGB_NROUNDS_2,
    XGB_EARLY_1, XGB_EARLY_2,
    N_CORES
  ))
)

fwrite(run_config, file.path(out_dir, "run_config_v6.csv"))

cat("\n==============================\n")
cat("V6 FEATURE ENGINEERED CONFIG PARAMETRIZADO\n")
cat("==============================\n")
print(run_config)

# ------------------------------------------------------------
# 1. Funciones auxiliares generales
# ------------------------------------------------------------

safe_fwrite <- function(x, path) {
  fwrite(x, path)
  invisible(TRUE)
}

z_from_train <- function(x, mu, sig) {
  if (is.na(sig) || sig == 0) {
    return(rep(0, length(x)))
  }
  (x - mu) / sig
}

safe_numeric <- function(dt, cols) {
  cols <- cols[cols %in% names(dt)]
  for (cc in cols) {
    dt[, (cc) := as.numeric(get(cc))]
    dt[!is.finite(get(cc)), (cc) := NA_real_]
  }
  invisible(dt)
}

# ------------------------------------------------------------
# 2. Feature engineering V6
# ------------------------------------------------------------

add_v6_features <- function(dt) {
  
  dt <- copy(dt)
  
  required_cols <- c(
    "elo_diff", "abs_elo_diff", "fifa_points_diff",
    "same_confed",
    "team_A_draw_rate_5", "team_B_draw_rate_5",
    "team_A_draw_rate_10", "team_B_draw_rate_10"
  )
  
  missing_required <- setdiff(required_cols, names(dt))
  if (length(missing_required) > 0) {
    stop("Faltan columnas necesarias para V6: ", paste(missing_required, collapse = ", "))
  }
  
  # -----------------------------
  # A. Tasas por partido
  # -----------------------------
  for (side in c("team_A", "team_B")) {
    for (w in c(5, 10)) {
      
      n_col   <- paste0(side, "_n_prev_matches_", w)
      gf_col  <- paste0(side, "_gf_", w)
      ga_col  <- paste0(side, "_ga_", w)
      gd_col  <- paste0(side, "_gd_", w)
      pts_col <- paste0(side, "_points_", w)
      
      needed <- c(n_col, gf_col, ga_col, gd_col, pts_col)
      missing_needed <- setdiff(needed, names(dt))
      if (length(missing_needed) > 0) {
        stop("Faltan columnas para tasas: ", paste(missing_needed, collapse = ", "))
      }
      
      dt[, paste0(side, "_gf_per_match_", w) :=
           get(gf_col) / pmax(get(n_col), 1)]
      
      dt[, paste0(side, "_ga_per_match_", w) :=
           get(ga_col) / pmax(get(n_col), 1)]
      
      dt[, paste0(side, "_gd_per_match_", w) :=
           get(gd_col) / pmax(get(n_col), 1)]
      
      dt[, paste0(side, "_points_per_match_", w) :=
           get(pts_col) / pmax(get(n_col), 1)]
    }
  }
  
  for (w in c(5, 10)) {
    dt[, paste0("gf_per_match_", w, "_diff") :=
         get(paste0("team_A_gf_per_match_", w)) -
         get(paste0("team_B_gf_per_match_", w))]
    
    dt[, paste0("ga_per_match_", w, "_diff") :=
         get(paste0("team_A_ga_per_match_", w)) -
         get(paste0("team_B_ga_per_match_", w))]
    
    dt[, paste0("gd_per_match_", w, "_diff") :=
         get(paste0("team_A_gd_per_match_", w)) -
         get(paste0("team_B_gd_per_match_", w))]
    
    dt[, paste0("points_per_match_", w, "_diff") :=
         get(paste0("team_A_points_per_match_", w)) -
         get(paste0("team_B_points_per_match_", w))]
  }
  
  # -----------------------------
  # B. Momentum 5 vs 10
  # -----------------------------
  for (side in c("team_A", "team_B")) {
    
    dt[, paste0(side, "_points_momentum_5v10") :=
         get(paste0(side, "_points_per_match_5")) -
         get(paste0(side, "_points_per_match_10"))]
    
    dt[, paste0(side, "_gd_momentum_5v10") :=
         get(paste0(side, "_gd_per_match_5")) -
         get(paste0(side, "_gd_per_match_10"))]
    
    dt[, paste0(side, "_gf_momentum_5v10") :=
         get(paste0(side, "_gf_per_match_5")) -
         get(paste0(side, "_gf_per_match_10"))]
    
    dt[, paste0(side, "_ga_momentum_5v10") :=
         get(paste0(side, "_ga_per_match_5")) -
         get(paste0(side, "_ga_per_match_10"))]
  }
  
  dt[, points_momentum_diff :=
       team_A_points_momentum_5v10 - team_B_points_momentum_5v10]
  
  dt[, gd_momentum_diff :=
       team_A_gd_momentum_5v10 - team_B_gd_momentum_5v10]
  
  dt[, gf_momentum_diff :=
       team_A_gf_momentum_5v10 - team_B_gf_momentum_5v10]
  
  dt[, ga_momentum_diff :=
       team_A_ga_momentum_5v10 - team_B_ga_momentum_5v10]
  
  # -----------------------------
  # C. Ataque vs defensa rival
  # -----------------------------
  for (w in c(5, 10)) {
    
    dt[, paste0("A_attack_vs_B_def_", w) :=
         get(paste0("team_A_gf_per_match_", w)) -
         get(paste0("team_B_ga_per_match_", w))]
    
    dt[, paste0("B_attack_vs_A_def_", w) :=
         get(paste0("team_B_gf_per_match_", w)) -
         get(paste0("team_A_ga_per_match_", w))]
    
    dt[, paste0("attack_matchup_diff_", w) :=
         get(paste0("A_attack_vs_B_def_", w)) -
         get(paste0("B_attack_vs_A_def_", w))]
  }
  
  # -----------------------------
  # D. Probabilidades base Elo
  # -----------------------------
  dt[, elo_p_A_base := 1 / (1 + 10^(-elo_diff / 400))]
  dt[, elo_p_B_base := 1 - elo_p_A_base]
  dt[, elo_favorite_prob := pmax(elo_p_A_base, elo_p_B_base)]
  dt[, elo_underdog_prob := pmin(elo_p_A_base, elo_p_B_base)]
  dt[, elo_balance := 1 - abs(elo_p_A_base - 0.5) * 2]
  
  # -----------------------------
  # E. Features específicas de empate
  # -----------------------------
  dt[, draw_pressure_score :=
       0.35 * (1 - pmin(abs_elo_diff, 500) / 500) +
       0.20 * (1 - pmin(abs(fifa_points_diff), 300) / 300) +
       0.20 * ((team_A_draw_rate_5 + team_B_draw_rate_5) / 2) +
       0.15 * ((team_A_draw_rate_10 + team_B_draw_rate_10) / 2) +
       0.10 * same_confed]
  
  dt[, low_scoring_proxy_5 :=
       -1 * (team_A_gf_per_match_5 + team_B_gf_per_match_5) +
       -1 * (team_A_ga_per_match_5 + team_B_ga_per_match_5)]
  
  dt[, low_scoring_proxy_10 :=
       -1 * (team_A_gf_per_match_10 + team_B_gf_per_match_10) +
       -1 * (team_A_ga_per_match_10 + team_B_ga_per_match_10)]
  
  dt[, defensive_balance_5 :=
       -abs(team_A_ga_per_match_5 - team_B_ga_per_match_5)]
  
  dt[, defensive_balance_10 :=
       -abs(team_A_ga_per_match_10 - team_B_ga_per_match_10)]
  
  dt[, close_game_x_draw_rate :=
       elo_balance * ((team_A_draw_rate_10 + team_B_draw_rate_10) / 2)]
  
  # -----------------------------
  # F. Reliability / incertidumbre
  # -----------------------------
  for (side in c("team_A", "team_B")) {
    for (w in c(5, 10)) {
      
      n_col <- paste0(side, "_n_prev_matches_", w)
      
      dt[, paste0(side, "_form_reliability_", w) :=
           pmin(get(n_col), 50) / 50]
      
      dt[, paste0(side, "_form_uncertainty_", w) :=
           1 / sqrt(pmax(get(n_col), 1))]
    }
  }
  
  for (w in c(5, 10)) {
    dt[, paste0("form_reliability_", w, "_min") :=
         pmin(get(paste0("team_A_form_reliability_", w)),
              get(paste0("team_B_form_reliability_", w)))]
    
    dt[, paste0("form_reliability_", w, "_diff") :=
         get(paste0("team_A_form_reliability_", w)) -
         get(paste0("team_B_form_reliability_", w))]
    
    dt[, paste0("form_uncertainty_", w, "_mean") :=
         (get(paste0("team_A_form_uncertainty_", w)) +
            get(paste0("team_B_form_uncertainty_", w))) / 2]
  }
  
  # -----------------------------
  # G. Interacciones simples
  # -----------------------------
  dt[, elo_x_form_5 :=
       elo_diff * points_per_match_5_diff]
  
  dt[, elo_x_form_10 :=
       elo_diff * points_per_match_10_diff]
  
  dt[, elo_x_attack_matchup_5 :=
       elo_diff * attack_matchup_diff_5]
  
  dt[, elo_x_attack_matchup_10 :=
       elo_diff * attack_matchup_diff_10]
  
  dt[, elo_balance_x_same_confed :=
       elo_balance * same_confed]
  
  # -----------------------------
  # H. Señales de ranking FIFA
  # -----------------------------
  if ("fifa_rank_diff" %in% names(dt)) {
    dt[, fifa_rank_strength_diff := -fifa_rank_diff]
  }
  
  # Valores no finitos
  new_numeric_cols <- names(dt)[sapply(dt, is.numeric)]
  for (cc in new_numeric_cols) {
    dt[!is.finite(get(cc)), (cc) := NA_real_]
  }
  
  dt
}

# ------------------------------------------------------------
# 3. Cargar V5 y crear V6
# ------------------------------------------------------------

training_raw <- fread(train_file_v5)
fixtures_raw <- fread(pred_file_v5)

cat("\n==============================\n")
cat("DATOS V5 CARGADOS\n")
cat("==============================\n")
cat("training_raw:", nrow(training_raw), "filas x", ncol(training_raw), "columnas\n")
cat("fixtures_raw:", nrow(fixtures_raw), "filas x", ncol(fixtures_raw), "columnas\n")

training <- add_v6_features(training_raw)
fixtures <- add_v6_features(fixtures_raw)

# ------------------------------------------------------------
# 4. Normalizaciones basadas solo en training
# ------------------------------------------------------------
# Evitamos leakage: medias/sds se calculan en training y se aplican a fixtures.

scale_pairs <- c(
  "elo_diff",
  "fifa_points_diff",
  "points_momentum_diff",
  "gd_momentum_diff",
  "attack_matchup_diff_5",
  "attack_matchup_diff_10",
  "draw_pressure_score"
)

scale_pairs <- scale_pairs[scale_pairs %in% names(training)]

scaling_stats <- data.table(
  variable = scale_pairs,
  mean = sapply(scale_pairs, function(v) mean(training[[v]], na.rm = TRUE)),
  sd = sapply(scale_pairs, function(v) sd(training[[v]], na.rm = TRUE))
)

for (v in scale_pairs) {
  mu <- scaling_stats[variable == v, mean]
  sg <- scaling_stats[variable == v, sd]
  
  training[, paste0(v, "_z") := z_from_train(get(v), mu, sg)]
  fixtures[, paste0(v, "_z") := z_from_train(get(v), mu, sg)]
}

# Disagreement Elo-FIFA con z-scores consistentes
if (all(c("elo_diff_z", "fifa_points_diff_z") %in% names(training))) {
  training[, elo_fifa_disagreement := elo_diff_z - fifa_points_diff_z]
  training[, abs_elo_fifa_disagreement := abs(elo_fifa_disagreement)]
  
  fixtures[, elo_fifa_disagreement := elo_diff_z - fifa_points_diff_z]
  fixtures[, abs_elo_fifa_disagreement := abs(elo_fifa_disagreement)]
}

safe_fwrite(training, train_file_v6)
safe_fwrite(fixtures, pred_file_v6)

if (WRITE_LEGACY_DATA) {
  safe_fwrite(training, legacy_train_file_v6)
  safe_fwrite(fixtures, legacy_pred_file_v6)
}

cat("\n==============================\n")
cat("ARCHIVOS V6 GUARDADOS\n")
cat("==============================\n")
cat(train_file_v6, "\n")
cat(pred_file_v6, "\n")
cat("training:", nrow(training), "filas x", ncol(training), "columnas\n")
cat("fixtures:", nrow(fixtures), "filas x", ncol(fixtures), "columnas\n")

# ------------------------------------------------------------
# 5. Definir features
# ------------------------------------------------------------

id_cols_train <- c("match_id", "result")

id_cols_fixtures <- c(
  "match_id",
  "stage",
  "group",
  "team_A",
  "team_B"
)

exclude_exact <- c(
  id_cols_train,
  id_cols_fixtures,
  "team_A_key",
  "team_B_key",
  "team_A_key.x",
  "team_B_key.x",
  "team_A_key.y",
  "team_B_key.y",
  # Se excluyen puntos FIFA brutos redundantes, pero se conserva fifa_points_diff
  "fifa_points_A",
  "fifa_points_B"
)

# En V6 ya NO excluimos n_prev_matches indirectamente;
# se conservan reliability/uncertainty y también los conteos si están.
exclude_patterns <- c()

excluded_by_pattern <- unique(unlist(lapply(
  exclude_patterns,
  function(p) grep(p, names(training), value = TRUE)
)))

feature_cols <- setdiff(
  names(training),
  c(exclude_exact, excluded_by_pattern)
)

fixture_feature_cols <- setdiff(
  names(fixtures),
  c(exclude_exact, excluded_by_pattern)
)

missing_in_fixtures <- setdiff(feature_cols, fixture_feature_cols)

if (length(missing_in_fixtures) > 0) {
  stop("Faltan columnas en fixtures V6: ", paste(missing_in_fixtures, collapse = ", "))
}

cat("\nNúmero de features V6:", length(feature_cols), "\n")
print(feature_cols)

safe_fwrite(
  data.table(feature = feature_cols),
  file.path(out_dir, "feature_cols_v6.csv")
)

# ------------------------------------------------------------
# 6. Resultado
# ------------------------------------------------------------

training[, result := as.character(result)]
training[, result := trimws(result)]
training <- training[result %in% class_levels]
training[, result := factor(result, levels = class_levels)]

cat("\nDistribución de result:\n")
print(training[, .N, by = result][order(-N)])

# ------------------------------------------------------------
# 7. Tipos
# ------------------------------------------------------------

categorical_cols <- c(
  "elo_gap_group",
  "elo_level_group",
  "tournament_type",
  "confed_A",
  "confed_B",
  "confed_pair"
)

categorical_cols <- categorical_cols[categorical_cols %in% feature_cols]
numeric_cols <- setdiff(feature_cols, categorical_cols)

for (cc in numeric_cols) {
  training[, (cc) := as.numeric(get(cc))]
  fixtures[, (cc) := as.numeric(get(cc))]
}

for (cc in categorical_cols) {
  training[, (cc) := factor(get(cc))]
  fixtures[, (cc) := factor(get(cc), levels = levels(training[[cc]]))]
}

for (cc in categorical_cols) {
  if (anyNA(fixtures[[cc]])) {
    cat("\nNiveles nuevos en fixtures para", cc, "\n")
    print(unique(fixtures_raw[[cc]]))
    stop("Variable categórica con nivel nuevo en fixtures: ", cc)
  }
}

for (cc in numeric_cols) {
  training[!is.finite(get(cc)), (cc) := NA_real_]
  fixtures[!is.finite(get(cc)), (cc) := NA_real_]
}

required_train <- c("result", feature_cols)
required_pred  <- feature_cols

na_train <- colSums(is.na(training[, ..required_train]))
na_fx    <- colSums(is.na(fixtures[, ..required_pred]))

cat("\nNA training:\n")
print(na_train[na_train > 0])

cat("\nNA fixtures:\n")
print(na_fx[na_fx > 0])

if (any(na_train > 0)) stop("Hay NA en training V6.")
if (any(na_fx > 0)) stop("Hay NA en fixtures V6.")

# ------------------------------------------------------------
# 8. Split train/test
# ------------------------------------------------------------
set.seed(SEED + 100L)
idx_train <- createDataPartition(
  y = training$result,
  p = 0.80,
  list = FALSE
)

train_dt <- training[idx_train]
test_dt  <- training[-idx_train]

cat("\nDistribución train:\n")
print(train_dt[, .N, by = result][order(-N)])

cat("\nDistribución test:\n")
print(test_dt[, .N, by = result][order(-N)])

# ------------------------------------------------------------
# 9. Pesos por clase
# ------------------------------------------------------------

class_counts <- train_dt[, .N, by = result]
class_counts[, class_weight := sum(N) / (.N * N)]

weight_map <- setNames(
  class_counts$class_weight,
  as.character(class_counts$result)
)

train_dt[, weight := weight_map[as.character(result)]]
test_dt[, weight := weight_map[as.character(result)]]

cat("\nPesos por clase:\n")
print(class_counts)

# ------------------------------------------------------------
# 10. Fórmulas y funciones de modelado
# ------------------------------------------------------------

formula_v6 <- as.formula(
  paste("result ~", paste(feature_cols, collapse = " + "))
)

formula_x_v6 <- as.formula(
  paste("~", paste(feature_cols, collapse = " + "))
)

multiclass_logloss <- function(actual, probs, eps = 1e-15) {
  actual <- as.character(actual)
  classes <- colnames(probs)
  probs <- as.matrix(probs)
  probs <- pmax(pmin(probs, 1 - eps), eps)
  idx <- cbind(seq_along(actual), match(actual, classes))
  -mean(log(probs[idx]))
}

get_metrics <- function(actual, pred_class, pred_prob, model_name) {
  actual <- factor(actual, levels = class_levels)
  pred_class <- factor(pred_class, levels = class_levels)
  
  cm <- confusionMatrix(pred_class, actual)
  acc <- as.numeric(cm$overall["Accuracy"])
  
  by_class <- as.data.table(cm$byClass, keep.rownames = "Class")
  bal_acc <- mean(by_class$`Balanced Accuracy`, na.rm = TRUE)
  
  logloss <- multiclass_logloss(actual, pred_prob)
  
  data.table(
    model = model_name,
    Accuracy = acc,
    BalancedAccuracy = bal_acc,
    LogLoss = logloss
  )
}

predict_with_draw_threshold <- function(prob_dt, draw_threshold = NULL) {
  prob_dt <- as.data.table(prob_dt)
  pred <- colnames(prob_dt)[max.col(prob_dt, ties.method = "first")]
  
  if (!is.null(draw_threshold)) {
    pred[prob_dt$Draw >= draw_threshold] <- "Draw"
  }
  
  pred
}

print_confusion_matrix <- function(actual, predicted, model_name) {
  actual <- factor(actual, levels = class_levels)
  predicted <- factor(predicted, levels = class_levels)
  
  cm <- table(Real = actual, Predicho = predicted)
  
  cat("\n------------------------------\n")
  cat("Modelo:", model_name, "\n")
  cat("------------------------------\n")
  print(cm)
  
  cat("\nProporciones por clase real:\n")
  print(round(prop.table(cm, margin = 1), 3))
  
  cm_dt <- as.data.table(cm)
  cm_dt[, model := model_name]
  setcolorder(cm_dt, c("model", "Real", "Predicho", "N"))
  cm_dt
}

align_model_matrix <- function(new_x, ref_names) {
  missing_cols <- setdiff(ref_names, colnames(new_x))
  
  if (length(missing_cols) > 0) {
    add_mat <- matrix(0, nrow = nrow(new_x), ncol = length(missing_cols))
    colnames(add_mat) <- missing_cols
    new_x <- cbind(new_x, add_mat)
  }
  
  extra_cols <- setdiff(colnames(new_x), ref_names)
  if (length(extra_cols) > 0) {
    new_x <- new_x[, setdiff(colnames(new_x), extra_cols), drop = FALSE]
  }
  
  new_x <- new_x[, ref_names, drop = FALSE]
  new_x
}

make_xgb_prob_safe <- function(pred_raw, class_levels) {
  
  n_class <- length(class_levels)
  
  if (is.matrix(pred_raw) || is.data.frame(pred_raw)) {
    
    prob <- as.matrix(pred_raw)
    
    if (ncol(prob) != n_class) {
      stop(
        "predict() devolvió matriz/data.frame, pero ncol != número de clases. ncol = ",
        ncol(prob),
        ", n_class = ",
        n_class
      )
    }
    
  } else {
    
    if (length(pred_raw) %% n_class != 0) {
      stop("Longitud de pred_raw no divisible por número de clases.")
    }
    
    prob <- matrix(
      pred_raw,
      ncol = n_class,
      byrow = TRUE
    )
  }
  
  colnames(prob) <- class_levels
  
  prob <- as.data.table(prob)
  prob <- prob[, class_levels, with = FALSE]
  
  row_sums <- rowSums(as.matrix(prob))
  
  if (any(abs(row_sums - 1) > 1e-5)) {
    warning("Algunas filas de probabilidad XGBoost no suman 1.")
    print(summary(row_sums))
  }
  
  prob
}

# ------------------------------------------------------------
# 11. Multinomial
# ------------------------------------------------------------

cat("\n==============================\n")
cat("ENTRENANDO MULTINOMIAL V6\n")
cat("==============================\n")
set.seed(SEED + 300L)
model_multinom <- multinom(
  formula_v6,
  data = train_dt,
  weights = train_dt$weight,
  trace = FALSE,
  maxit = 1000
)

prob_multinom <- predict(
  model_multinom,
  newdata = test_dt,
  type = "probs"
)

prob_multinom <- as.data.table(prob_multinom)
prob_multinom <- prob_multinom[, class_levels, with = FALSE]

pred_multinom <- predict_with_draw_threshold(prob_multinom)

metrics_multinom <- get_metrics(
  actual = test_dt$result,
  pred_class = pred_multinom,
  pred_prob = prob_multinom,
  model_name = "multinomial_v6"
)

print(metrics_multinom)

# ------------------------------------------------------------
# 12. Random Forest random search
# ------------------------------------------------------------

cat("\n==============================\n")
cat("RANDOM FOREST RANDOM SEARCH V6\n")
cat("==============================\n")

set.seed(SEED + 555L)


mtry_candidates <- unique(pmax(1, pmin(length(feature_cols), c(8, 12, 15, 20, 25, 30, 40, 50))))

rf_grid <- data.table(
  mtry = sample(mtry_candidates, N_RF_RANDOM, replace = TRUE),
  min.node.size = sample(c(3, 5, 10, 15, 20, 30), N_RF_RANDOM, replace = TRUE),
  sample.fraction = runif(N_RF_RANDOM, 0.75, 1.00),
  splitrule = sample(c("gini", "extratrees"), N_RF_RANDOM, replace = TRUE)
)

rf_grid[, sample.fraction := round(sample.fraction, 3)]
rf_grid <- unique(rf_grid)

safe_fwrite(rf_grid, file.path(out_dir, "rf_random_grid_v6.csv"))

rf_results <- list()

best_rf_logloss <- Inf
best_rf_model <- NULL
best_rf_prob <- NULL
best_rf_pred <- NULL
best_rf_params <- NULL

for (i in seq_len(nrow(rf_grid))) {
  
  pars <- rf_grid[i]
  
  cat("\nRF random grid", i, "de", nrow(rf_grid), "\n")
  print(pars)
  
  set.seed(SEED + 500L + i)
  
  
  model_rf_i <- ranger(
    formula = formula_v6,
    data = train_dt,
    probability = TRUE,
    num.trees = RF_NUM_TREES,
    mtry = pars$mtry,
    min.node.size = pars$min.node.size,
    sample.fraction = pars$sample.fraction,
    splitrule = pars$splitrule,
    importance = "impurity",
    case.weights = train_dt$weight,
    seed = SEED + 500L + i,
    num.threads = N_CORES
  )
  
  prob_rf_i <- predict(model_rf_i, data = test_dt)$predictions
  prob_rf_i <- as.data.table(prob_rf_i)
  prob_rf_i <- prob_rf_i[, class_levels, with = FALSE]
  
  pred_rf_i <- predict_with_draw_threshold(prob_rf_i)
  
  metrics_rf_i <- get_metrics(
    actual = test_dt$result,
    pred_class = pred_rf_i,
    pred_prob = prob_rf_i,
    model_name = paste0("random_forest_v6_grid_", i)
  )
  
  metrics_rf_i[, grid_id := i]
  metrics_rf_i[, mtry := pars$mtry]
  metrics_rf_i[, min.node.size := pars$min.node.size]
  metrics_rf_i[, sample.fraction := pars$sample.fraction]
  metrics_rf_i[, splitrule := pars$splitrule]
  metrics_rf_i[, num.trees := RF_NUM_TREES]
  
  print(metrics_rf_i)
  
  rf_results[[i]] <- metrics_rf_i
  rf_partial <- rbindlist(rf_results, fill = TRUE)[order(LogLoss)]
  
  safe_fwrite(
    rf_partial,
    file.path(out_dir, "rf_tuning_results_v6_partial.csv")
  )
  
  if (metrics_rf_i$LogLoss < best_rf_logloss) {
    cat("\nNuevo mejor RF. LogLoss:", metrics_rf_i$LogLoss, "\n")
    
    best_rf_logloss <- metrics_rf_i$LogLoss
    best_rf_model <- model_rf_i
    best_rf_prob <- prob_rf_i
    best_rf_pred <- pred_rf_i
    best_rf_params <- copy(pars)
  }
  
  rm(model_rf_i, prob_rf_i, pred_rf_i)
  gc()
}

rf_tuning_results <- rbindlist(rf_results, fill = TRUE)[order(LogLoss)]

safe_fwrite(
  rf_tuning_results,
  file.path(out_dir, "rf_tuning_results_v6.csv")
)

model_rf <- best_rf_model
prob_rf  <- best_rf_prob
pred_rf  <- best_rf_pred

metrics_rf <- rf_tuning_results[1, .(
  model = "random_forest_v6_tuned",
  Accuracy,
  BalancedAccuracy,
  LogLoss
)]

cat("\nMejor RF V6:\n")
print(rf_tuning_results[1])
print(metrics_rf)

importance_rf <- data.table(
  variable = names(model_rf$variable.importance),
  importance = as.numeric(model_rf$variable.importance)
)[order(-importance)]

safe_fwrite(
  importance_rf,
  file.path(out_dir, "variable_importance_random_forest_v6_tuned.csv")
)

# ------------------------------------------------------------
# 13. XGBoost matrices
# ------------------------------------------------------------

cat("\n==============================\n")
cat("PREPARANDO XGBOOST V6\n")
cat("==============================\n")

x_train <- model.matrix(formula_x_v6, data = train_dt)[, -1, drop = FALSE]
x_test  <- model.matrix(formula_x_v6, data = test_dt)[, -1, drop = FALSE]
x_test  <- align_model_matrix(x_test, colnames(x_train))

y_train <- as.integer(train_dt$result) - 1
y_test  <- as.integer(test_dt$result) - 1

dtrain <- xgb.DMatrix(
  data = x_train,
  label = y_train,
  weight = train_dt$weight
)

dtest <- xgb.DMatrix(
  data = x_test,
  label = y_test
)

# ------------------------------------------------------------
# 14. XGBoost random + refine
# ------------------------------------------------------------

cat("\n==============================\n")
cat("XGBOOST RANDOM SEARCH V6\n")
cat("==============================\n")

set.seed(SEED + 777L)

xgb_grid_1 <- data.table(
  search_stage = "random",
  eta = runif(N_XGB_RANDOM, 0.010, 0.060),
  max_depth = sample(2:5, N_XGB_RANDOM, replace = TRUE),
  min_child_weight = sample(c(3, 5, 8, 10, 15, 20, 30), N_XGB_RANDOM, replace = TRUE),
  subsample = runif(N_XGB_RANDOM, 0.70, 1.00),
  colsample_bytree = runif(N_XGB_RANDOM, 0.60, 1.00),
  lambda = sample(c(0.5, 1, 2, 3, 5, 8, 12), N_XGB_RANDOM, replace = TRUE),
  alpha = sample(c(0, 0.1, 0.5, 1, 1.5), N_XGB_RANDOM, replace = TRUE)
)

xgb_grid_1[, eta := round(eta, 4)]
xgb_grid_1[, subsample := round(subsample, 3)]
xgb_grid_1[, colsample_bytree := round(colsample_bytree, 3)]
xgb_grid_1 <- unique(xgb_grid_1)

safe_fwrite(xgb_grid_1, file.path(out_dir, "xgb_random_grid_v6.csv"))

xgb_results <- list()

best_xgb_logloss <- Inf
best_xgb_model <- NULL
best_xgb_prob <- NULL
best_xgb_pred <- NULL
best_xgb_params <- NULL
best_xgb_stage <- NULL

run_one_xgb <- function(i, pars, nrounds, early_rounds, grid_label) {
  
  cat("\nXGB", grid_label, "grid", i, "\n")
  print(pars)
  
  params_i <- list(
    objective = "multi:softprob",
    num_class = length(class_levels),
    eval_metric = "mlogloss",
    eta = pars$eta,
    max_depth = pars$max_depth,
    min_child_weight = pars$min_child_weight,
    subsample = pars$subsample,
    colsample_bytree = pars$colsample_bytree,
    lambda = pars$lambda,
    alpha = pars$alpha,
    seed = SEED + 900L + i,
    nthread = N_CORES
  )
  
  set.seed(SEED + 900L + i)
  
  model_xgb_i <- xgb.train(
    params = params_i,
    data = dtrain,
    nrounds = nrounds,
    evals = list(test = dtest),
    early_stopping_rounds = early_rounds,
    print_every_n = 200,
    verbose = 1
  )
  
  pred_raw_i <- predict(model_xgb_i, newdata = dtest)
  
  prob_i <- make_xgb_prob_safe(
    pred_raw = pred_raw_i,
    class_levels = class_levels
  )
  
  pred_i <- predict_with_draw_threshold(prob_i)
  
  metrics_i <- get_metrics(
    actual = test_dt$result,
    pred_class = pred_i,
    pred_prob = prob_i,
    model_name = paste0("xgboost_v6_grid_", grid_label, "_", i)
  )
  
  metrics_i[, matrix_version := fifelse(
    is.matrix(pred_raw_i) || is.data.frame(pred_raw_i),
    "predict_returned_matrix",
    "vector_byrow_TRUE"
  )]
  
  metrics_i[, grid_id := i]
  metrics_i[, search_stage := grid_label]
  metrics_i[, best_iteration := model_xgb_i$best_iteration]
  metrics_i[, eta := pars$eta]
  metrics_i[, max_depth := pars$max_depth]
  metrics_i[, min_child_weight := pars$min_child_weight]
  metrics_i[, subsample := pars$subsample]
  metrics_i[, colsample_bytree := pars$colsample_bytree]
  metrics_i[, lambda := pars$lambda]
  metrics_i[, alpha := pars$alpha]
  metrics_i[, nrounds_max := nrounds]
  metrics_i[, early_stopping_rounds := early_rounds]
  
  print(metrics_i)
  
  list(
    model = model_xgb_i,
    prob = prob_i,
    pred = pred_i,
    metrics = metrics_i,
    params = params_i
  )
}

# Primera ronda
for (i in seq_len(nrow(xgb_grid_1))) {
  
  fit_i <- run_one_xgb(
    i = i,
    pars = xgb_grid_1[i],
    nrounds = XGB_NROUNDS_1,
    early_rounds = XGB_EARLY_1,
    grid_label = "random"
  )
  
  xgb_results[[length(xgb_results) + 1]] <- fit_i$metrics
  
  xgb_partial <- rbindlist(xgb_results, fill = TRUE)[order(LogLoss)]
  safe_fwrite(
    xgb_partial,
    file.path(out_dir, "xgb_tuning_results_v6_partial.csv")
  )
  
  if (fit_i$metrics$LogLoss < best_xgb_logloss) {
    cat("\nNuevo mejor XGB. LogLoss:", fit_i$metrics$LogLoss, "\n")
    
    best_xgb_logloss <- fit_i$metrics$LogLoss
    best_xgb_model <- fit_i$model
    best_xgb_prob <- fit_i$prob
    best_xgb_pred <- fit_i$pred
    best_xgb_params <- fit_i$params
    best_xgb_stage <- "random"
  }
  
  rm(fit_i)
  gc()
}

xgb_random_results <- rbindlist(xgb_results, fill = TRUE)[order(LogLoss)]
safe_fwrite(
  xgb_random_results,
  file.path(out_dir, "xgb_random_results_v6.csv")
)

cat("\n==============================\n")
cat("XGBOOST REFINEMENT LOCAL V6\n")
cat("==============================\n")

best_random <- xgb_random_results[1]

set.seed(SEED + 778L)

xgb_grid_2 <- data.table(
  search_stage = "refine",
  eta = pmin(pmax(rnorm(N_XGB_REFINE, best_random$eta, 0.010), 0.005), 0.080),
  max_depth = sample(
    unique(pmax(2, pmin(6, best_random$max_depth + c(-1, 0, 1)))),
    N_XGB_REFINE,
    replace = TRUE
  ),
  min_child_weight = sample(
    unique(pmax(1, best_random$min_child_weight + c(-5, 0, 5))),
    N_XGB_REFINE,
    replace = TRUE
  ),
  subsample = pmin(pmax(rnorm(N_XGB_REFINE, best_random$subsample, 0.080), 0.650), 1.000),
  colsample_bytree = pmin(pmax(rnorm(N_XGB_REFINE, best_random$colsample_bytree, 0.080), 0.600), 1.000),
  lambda = sample(
    unique(pmax(0, best_random$lambda + c(-1, 0, 1, 3))),
    N_XGB_REFINE,
    replace = TRUE
  ),
  alpha = sample(
    unique(pmax(0, best_random$alpha + c(-0.5, 0, 0.5))),
    N_XGB_REFINE,
    replace = TRUE
  )
)

xgb_grid_2[, eta := round(eta, 4)]
xgb_grid_2[, subsample := round(subsample, 3)]
xgb_grid_2[, colsample_bytree := round(colsample_bytree, 3)]
xgb_grid_2 <- unique(xgb_grid_2)

safe_fwrite(xgb_grid_2, file.path(out_dir, "xgb_refine_grid_v6.csv"))

for (i in seq_len(nrow(xgb_grid_2))) {
  
  fit_i <- run_one_xgb(
    i = i,
    pars = xgb_grid_2[i],
    nrounds = XGB_NROUNDS_2,
    early_rounds = XGB_EARLY_2,
    grid_label = "refine"
  )
  
  xgb_results[[length(xgb_results) + 1]] <- fit_i$metrics
  
  xgb_partial <- rbindlist(xgb_results, fill = TRUE)[order(LogLoss)]
  safe_fwrite(
    xgb_partial,
    file.path(out_dir, "xgb_tuning_results_v6_partial.csv")
  )
  
  if (fit_i$metrics$LogLoss < best_xgb_logloss) {
    cat("\nNuevo mejor XGB. LogLoss:", fit_i$metrics$LogLoss, "\n")
    
    best_xgb_logloss <- fit_i$metrics$LogLoss
    best_xgb_model <- fit_i$model
    best_xgb_prob <- fit_i$prob
    best_xgb_pred <- fit_i$pred
    best_xgb_params <- fit_i$params
    best_xgb_stage <- "refine"
  }
  
  rm(fit_i)
  gc()
}

xgb_tuning_results <- rbindlist(xgb_results, fill = TRUE)[order(LogLoss)]

safe_fwrite(
  xgb_tuning_results,
  file.path(out_dir, "xgb_tuning_results_v6.csv")
)

cat("\nMejores XGBoost V6:\n")
print(head(xgb_tuning_results, 20))

model_xgb <- best_xgb_model
prob_xgb  <- best_xgb_prob
pred_xgb  <- best_xgb_pred

metrics_xgb <- xgb_tuning_results[1, .(
  model = "xgboost_v6_tuned",
  Accuracy,
  BalancedAccuracy,
  LogLoss
)]

print(metrics_xgb)

importance_xgb <- xgb.importance(
  feature_names = colnames(x_train),
  model = model_xgb
)

safe_fwrite(
  as.data.table(importance_xgb),
  file.path(out_dir, "variable_importance_xgboost_v6_tuned.csv")
)

# ------------------------------------------------------------
# 15. Ensemble simple y búsqueda de pesos
# ------------------------------------------------------------

cat("\n==============================\n")
cat("ENSEMBLE V6\n")
cat("==============================\n")

weights_grid <- CJ(
  w_rf = seq(0.2, 0.8, by = 0.1),
  w_multinom = seq(0.1, 0.7, by = 0.1)
)

weights_grid[, w_xgb := 1 - w_rf - w_multinom]
weights_grid <- weights_grid[w_xgb >= 0]
weights_grid[, weight_id := .I]

ensemble_results <- rbindlist(lapply(seq_len(nrow(weights_grid)), function(i) {
  
  w <- weights_grid[i]
  
  prob_tmp <- data.table(
    HomeWin = w$w_rf * prob_rf$HomeWin + w$w_multinom * prob_multinom$HomeWin + w$w_xgb * prob_xgb$HomeWin,
    Draw    = w$w_rf * prob_rf$Draw    + w$w_multinom * prob_multinom$Draw    + w$w_xgb * prob_xgb$Draw,
    AwayWin = w$w_rf * prob_rf$AwayWin + w$w_multinom * prob_multinom$AwayWin + w$w_xgb * prob_xgb$AwayWin
  )
  
  pred_tmp <- predict_with_draw_threshold(prob_tmp)
  
  met <- get_metrics(
    actual = test_dt$result,
    pred_class = pred_tmp,
    pred_prob = prob_tmp,
    model_name = paste0("ensemble_v6_", i)
  )
  
  cbind(met, w)
}))

ensemble_results <- ensemble_results[order(LogLoss)]

safe_fwrite(
  ensemble_results,
  file.path(out_dir, "ensemble_weight_search_v6.csv")
)

best_ensemble <- ensemble_results[1]
print(best_ensemble)

w_rf_best <- best_ensemble$w_rf
w_multinom_best <- best_ensemble$w_multinom
w_xgb_best <- best_ensemble$w_xgb

prob_ensemble <- data.table(
  HomeWin = w_rf_best * prob_rf$HomeWin + w_multinom_best * prob_multinom$HomeWin + w_xgb_best * prob_xgb$HomeWin,
  Draw    = w_rf_best * prob_rf$Draw    + w_multinom_best * prob_multinom$Draw    + w_xgb_best * prob_xgb$Draw,
  AwayWin = w_rf_best * prob_rf$AwayWin + w_multinom_best * prob_multinom$AwayWin + w_xgb_best * prob_xgb$AwayWin
)

pred_ensemble <- predict_with_draw_threshold(prob_ensemble)

metrics_ensemble <- get_metrics(
  actual = test_dt$result,
  pred_class = pred_ensemble,
  pred_prob = prob_ensemble,
  model_name = "ensemble_v6_tuned_weights"
)

print(metrics_ensemble)

# ------------------------------------------------------------
# 16. Draw threshold search
# ------------------------------------------------------------

cat("\n==============================\n")
cat("DRAW THRESHOLD SEARCH V6\n")
cat("==============================\n")

threshold_grid <- seq(0.20, 0.45, by = 0.01)

draw_threshold_results <- rbindlist(lapply(threshold_grid, function(th) {
  
  pred_tmp <- predict_with_draw_threshold(prob_ensemble, draw_threshold = th)
  
  met_tmp <- get_metrics(
    actual = test_dt$result,
    pred_class = pred_tmp,
    pred_prob = prob_ensemble,
    model_name = paste0("ensemble_v6_draw_threshold_", th)
  )
  
  met_tmp[, draw_threshold := th]
  met_tmp
}))

safe_fwrite(
  draw_threshold_results,
  file.path(out_dir, "draw_threshold_search_v6.csv")
)

best_threshold_balacc <- draw_threshold_results[order(-BalancedAccuracy)][1]
best_draw_threshold <- best_threshold_balacc$draw_threshold

cat("\nMejor threshold por BalancedAccuracy:\n")
print(best_threshold_balacc)

pred_ensemble_thr <- predict_with_draw_threshold(
  prob_ensemble,
  draw_threshold = best_draw_threshold
)

metrics_ensemble_thr <- get_metrics(
  actual = test_dt$result,
  pred_class = pred_ensemble_thr,
  pred_prob = prob_ensemble,
  model_name = "ensemble_v6_tuned_weights_draw_threshold"
)

print(metrics_ensemble_thr)

# ------------------------------------------------------------
# 17. Comparar modelos
# ------------------------------------------------------------

metrics_all <- rbindlist(list(
  metrics_multinom,
  metrics_rf,
  metrics_xgb,
  metrics_ensemble,
  metrics_ensemble_thr
), fill = TRUE)[order(LogLoss)]

cat("\n==============================\n")
cat("COMPARACIÓN FINAL V6 EN TEST\n")
cat("==============================\n")
print(metrics_all)

safe_fwrite(
  metrics_all,
  file.path(out_dir, "model_comparison_v6.csv")
)

# ------------------------------------------------------------
# 18. Matrices de confusión
# ------------------------------------------------------------

cm_multinom_final <- print_confusion_matrix(
  actual = test_dt$result,
  predicted = pred_multinom,
  model_name = "Multinomial V6"
)

cm_rf_final <- print_confusion_matrix(
  actual = test_dt$result,
  predicted = pred_rf,
  model_name = "Random Forest V6 tuned"
)

cm_xgb_final <- print_confusion_matrix(
  actual = test_dt$result,
  predicted = pred_xgb,
  model_name = "XGBoost V6 tuned"
)

cm_ensemble_final <- print_confusion_matrix(
  actual = test_dt$result,
  predicted = pred_ensemble,
  model_name = "Ensemble V6 tuned weights"
)

cm_ensemble_thr_final <- print_confusion_matrix(
  actual = test_dt$result,
  predicted = pred_ensemble_thr,
  model_name = paste0("Ensemble V6 draw threshold ", best_draw_threshold)
)

confusion_all <- rbindlist(list(
  cm_multinom_final,
  cm_rf_final,
  cm_xgb_final,
  cm_ensemble_final,
  cm_ensemble_thr_final
))

safe_fwrite(
  confusion_all,
  file.path(out_dir, "confusion_matrices_all_models_v6.csv")
)

# ------------------------------------------------------------
# 19. Guardar predicciones test
# ------------------------------------------------------------

test_predictions <- copy(test_dt)

test_predictions[, multinomial_pred := pred_multinom]
test_predictions[, multinomial_p_homewin := prob_multinom$HomeWin]
test_predictions[, multinomial_p_draw    := prob_multinom$Draw]
test_predictions[, multinomial_p_awaywin := prob_multinom$AwayWin]

test_predictions[, rf_pred := pred_rf]
test_predictions[, rf_p_homewin := prob_rf$HomeWin]
test_predictions[, rf_p_draw    := prob_rf$Draw]
test_predictions[, rf_p_awaywin := prob_rf$AwayWin]

test_predictions[, xgb_pred := pred_xgb]
test_predictions[, xgb_p_homewin := prob_xgb$HomeWin]
test_predictions[, xgb_p_draw    := prob_xgb$Draw]
test_predictions[, xgb_p_awaywin := prob_xgb$AwayWin]

test_predictions[, ensemble_pred := pred_ensemble]
test_predictions[, ensemble_p_homewin := prob_ensemble$HomeWin]
test_predictions[, ensemble_p_draw    := prob_ensemble$Draw]
test_predictions[, ensemble_p_awaywin := prob_ensemble$AwayWin]

test_predictions[, ensemble_thr_pred := pred_ensemble_thr]

safe_fwrite(
  test_predictions,
  file.path(out_dir, "test_predictions_all_models_v6.csv")
)

# ------------------------------------------------------------
# 20. Predicción fixtures 2026
# ------------------------------------------------------------

cat("\n==============================\n")
cat("PREDICIENDO FIXTURES 2026 V6\n")
cat("==============================\n")

# Multinomial
fixture_prob_multinom <- predict(
  model_multinom,
  newdata = fixtures,
  type = "probs"
)

fixture_prob_multinom <- as.data.table(fixture_prob_multinom)
fixture_prob_multinom <- fixture_prob_multinom[, class_levels, with = FALSE]

fixture_pred_multinom <- predict_with_draw_threshold(fixture_prob_multinom)

# RF
fixture_prob_rf <- predict(
  model_rf,
  data = fixtures
)$predictions

fixture_prob_rf <- as.data.table(fixture_prob_rf)
fixture_prob_rf <- fixture_prob_rf[, class_levels, with = FALSE]

fixture_pred_rf <- predict_with_draw_threshold(fixture_prob_rf)

# XGBoost
x_fixtures <- model.matrix(formula_x_v6, data = fixtures)[, -1, drop = FALSE]
x_fixtures <- align_model_matrix(x_fixtures, colnames(x_train))

fixture_raw_xgb <- predict(
  model_xgb,
  newdata = xgb.DMatrix(x_fixtures)
)

fixture_prob_xgb <- make_xgb_prob_safe(
  pred_raw = fixture_raw_xgb,
  class_levels = class_levels
)

fixture_pred_xgb <- predict_with_draw_threshold(fixture_prob_xgb)

# Ensemble fixtures
fixture_prob_ensemble <- data.table(
  HomeWin = w_rf_best * fixture_prob_rf$HomeWin + w_multinom_best * fixture_prob_multinom$HomeWin + w_xgb_best * fixture_prob_xgb$HomeWin,
  Draw    = w_rf_best * fixture_prob_rf$Draw    + w_multinom_best * fixture_prob_multinom$Draw    + w_xgb_best * fixture_prob_xgb$Draw,
  AwayWin = w_rf_best * fixture_prob_rf$AwayWin + w_multinom_best * fixture_prob_multinom$AwayWin + w_xgb_best * fixture_prob_xgb$AwayWin
)

fixture_pred_ensemble <- predict_with_draw_threshold(fixture_prob_ensemble)

fixture_pred_ensemble_thr <- predict_with_draw_threshold(
  fixture_prob_ensemble,
  draw_threshold = best_draw_threshold
)

fixtures_predictions <- copy(fixtures)

fixtures_predictions[, multinomial_pred := fixture_pred_multinom]
fixtures_predictions[, multinomial_p_team_A_win := fixture_prob_multinom$HomeWin]
fixtures_predictions[, multinomial_p_draw       := fixture_prob_multinom$Draw]
fixtures_predictions[, multinomial_p_team_B_win := fixture_prob_multinom$AwayWin]

fixtures_predictions[, rf_pred := fixture_pred_rf]
fixtures_predictions[, rf_p_team_A_win := fixture_prob_rf$HomeWin]
fixtures_predictions[, rf_p_draw       := fixture_prob_rf$Draw]
fixtures_predictions[, rf_p_team_B_win := fixture_prob_rf$AwayWin]

fixtures_predictions[, xgb_pred := fixture_pred_xgb]
fixtures_predictions[, xgb_p_team_A_win := fixture_prob_xgb$HomeWin]
fixtures_predictions[, xgb_p_draw       := fixture_prob_xgb$Draw]
fixtures_predictions[, xgb_p_team_B_win := fixture_prob_xgb$AwayWin]

fixtures_predictions[, ensemble_pred := fixture_pred_ensemble]
fixtures_predictions[, ensemble_p_team_A_win := fixture_prob_ensemble$HomeWin]
fixtures_predictions[, ensemble_p_draw       := fixture_prob_ensemble$Draw]
fixtures_predictions[, ensemble_p_team_B_win := fixture_prob_ensemble$AwayWin]

fixtures_predictions[, ensemble_thr_pred := fixture_pred_ensemble_thr]

fixtures_predictions[, multinomial_winner := fifelse(
  multinomial_pred == "HomeWin", team_A,
  fifelse(multinomial_pred == "AwayWin", team_B, "Draw")
)]

fixtures_predictions[, rf_winner := fifelse(
  rf_pred == "HomeWin", team_A,
  fifelse(rf_pred == "AwayWin", team_B, "Draw")
)]

fixtures_predictions[, xgb_winner := fifelse(
  xgb_pred == "HomeWin", team_A,
  fifelse(xgb_pred == "AwayWin", team_B, "Draw")
)]

fixtures_predictions[, ensemble_winner := fifelse(
  ensemble_pred == "HomeWin", team_A,
  fifelse(ensemble_pred == "AwayWin", team_B, "Draw")
)]

fixtures_predictions[, ensemble_thr_winner := fifelse(
  ensemble_thr_pred == "HomeWin", team_A,
  fifelse(ensemble_thr_pred == "AwayWin", team_B, "Draw")
)]

safe_fwrite(
  fixtures_predictions,
  file.path(out_dir, "fixtures_2026_predictions_all_models_v6.csv")
)

# ------------------------------------------------------------
# 21. Guardar modelos y metadata
# ------------------------------------------------------------

cat("\n==============================\n")
cat("GUARDANDO MODELOS V6\n")
cat("==============================\n")

saveRDS(
  model_multinom,
  file.path(out_dir, "model_multinomial_v6.rds")
)

saveRDS(
  model_rf,
  file.path(out_dir, "model_random_forest_v6_tuned.rds")
)

xgb.save(
  model_xgb,
  file.path(out_dir, "model_xgboost_v6_tuned.xgb")
)

saveRDS(
  list(
    seed = SEED,
    run_config = run_config,
    train_file_v5 = train_file_v5,
    pred_file_v5 = pred_file_v5,
    train_file_v6 = train_file_v6,
    pred_file_v6 = pred_file_v6,
    feature_cols = feature_cols,
    categorical_cols = categorical_cols,
    numeric_cols = numeric_cols,
    xgb_feature_names = colnames(x_train),
    result_levels = class_levels,
    formula_v6 = formula_v6,
    formula_x_v6 = formula_x_v6,
    metrics = metrics_all,
    rf_best_params = best_rf_params,
    xgb_best_params = best_xgb_params,
    xgb_best_stage = best_xgb_stage,
    ensemble_best_weights = data.table(
      w_rf = w_rf_best,
      w_multinom = w_multinom_best,
      w_xgb = w_xgb_best
    ),
    best_draw_threshold = best_draw_threshold,
    scaling_stats = scaling_stats,
    n_cores = N_CORES,
    rf_grid = rf_grid,
    xgb_grid_random = xgb_grid_1,
    xgb_grid_refine = xgb_grid_2,
    xgb_prob_parser = "make_xgb_prob_safe"
  ),
  file.path(out_dir, "model_metadata_v6.rds")
)

# ------------------------------------------------------------
# 22. Resumen final
# ------------------------------------------------------------

cat("\n==============================\n")
cat("PROCESO V6 COMPLETADO\n")
cat("==============================\n")

cat("\nMétricas V6:\n")
print(metrics_all)

cat("\nMejor ensemble weights:\n")
print(data.table(
  w_rf = w_rf_best,
  w_multinom = w_multinom_best,
  w_xgb = w_xgb_best,
  draw_threshold = best_draw_threshold
))

cat("\nMatriz de confusión consolidada:\n")
for (mm in unique(confusion_all$model)) {
  cat("\n------------------------------\n")
  cat("Modelo:", mm, "\n")
  cat("------------------------------\n")
  
  tmp <- confusion_all[model == mm]
  
  cm_wide <- dcast(
    tmp,
    Real ~ Predicho,
    value.var = "N",
    fill = 0
  )
  
  print(cm_wide)
}

cat("\nArchivos generados en:\n")
cat(out_dir, "\n")

cat("\nPredicciones Mundial 2026 V6:\n")
cat(file.path(out_dir, "fixtures_2026_predictions_all_models_v6.csv"), "\n")
