# ============================================================
# 05_mc_worldcup_v6_multiseed_annexC.R
# Monte Carlo V6 multiseed: grupos promedio 10 seeds + KO dinámico promedio 10 seeds + FIFA Annex C
# ============================================================
#
# Este script NO reentrena modelos.
#
# Lee salidas de V6:
#   modelos_v6_feature_engineered/model_metadata_v6.rds
#   modelos_v6_feature_engineered/model_multinomial_v6.rds
#   modelos_v6_feature_engineered/model_random_forest_v6_tuned.rds
#   modelos_v6_feature_engineered/model_xgboost_v6_tuned.xgb
#   modelos_v6_feature_engineered/fixtures_2026_predictions_all_models_v6.csv
#   datos/training_v6_feature_engineered.csv
#   datos/fixtures_2026_v6_feature_engineered.csv
#
# Qué mejora frente al V6B:
#   - En grupos usa las probabilidades ya predichas por V6.
#   - En KO NO usa solo fuerza aproximada.
#   - Para cada cruce KO simulado construye una nueva fila:
#       build_match_features(team_A, team_B)
#     y predice:
#       multinomial + RF + XGBoost + ensemble
#
# Nota crítica:
#   - El bracket de R32 es aproximado para los mejores terceros.
#   - Si luego tienes la tabla oficial de asignación de terceros FIFA 2026,
#     se puede reemplazar make_approx_r32_bracket().
#
# Salidas:
#   modelos_v6_feature_engineered/MONTE_CARLO_V6MS_OFFICIAL_ANNEXC/
#
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(nnet)
  library(ranger)
  library(xgboost)
})
# ------------------------------------------------------------
# 0. Configuración
# ------------------------------------------------------------

set.seed(2026)

# Entradas multiseed
base_multiseed_dir <- "resultados_server/v6_multiseed"
consolidated_dir   <- "resultados_server/v6_multiseed_consolidated"
mc_out_dir         <- file.path(consolidated_dir, "MONTE_CARLO_V6_MULTISEED_OFFICIAL_ANNEXC")

if (!dir.exists(mc_out_dir)) {
  dir.create(mc_out_dir, recursive = TRUE)
}

# Número de simulaciones. Para prueba usa 1000-5000; final puedes subir a 50000-100000.
N_SIM <- 50

# En grupos se usa el promedio multiseed ya consolidado.
GROUP_PROB_SOURCE <- "multiseed_mean"

# En KO se predice cada cruce con los 10 modelos y se promedian probabilidades.
KO_PROB_SOURCE <- "multiseed_mean_ensemble"

# Incertidumbre Dirichlet en grupos.
USE_DIRICHLET_GROUP <- TRUE
DIRICHLET_GROUP_CONCENTRATION <- 100

# Incertidumbre Dirichlet en KO sobre probabilidades de 90 minutos.
USE_DIRICHLET_KO <- TRUE
DIRICHLET_KO_CONCENTRATION <- 120

# En una simulación KO, si el partido sale empate en 90 min,
# se decide el avance por penales/prolongación usando ventaja suavizada.
PENALTY_SHRINK_MIN <- 0.35
PENALTY_SHRINK_MAX <- 0.65

# Hosts del Mundial 2026. Ajusta nombres si en tus datos tienen otra escritura.
HOST_TEAMS <- c("Mexico", "United States", "USA", "Canada")

# Archivo ya generado por 04_check_multiseed_v6.R
pred_file <- file.path(consolidated_dir, "multiseed_fixtures_probabilities_v6.csv")

# Detectar seeds completas
seed_dirs <- list.dirs(base_multiseed_dir, recursive = FALSE, full.names = TRUE)
seed_dirs <- seed_dirs[grepl("seed_[0-9]+$", seed_dirs)]
seed_info <- data.table(
  seed_dir = seed_dirs,
  seed = as.integer(sub(".*seed_", "", seed_dirs))
)[order(seed)]

if (nrow(seed_info) == 0) {
  stop("No se encontraron carpetas seed_* en: ", base_multiseed_dir)
}

required_seed_files <- c(
  "model_metadata_v6.rds",
  "model_multinomial_v6.rds",
  "model_random_forest_v6_tuned.rds",
  "model_xgboost_v6_tuned.xgb",
  "training_v6_feature_engineered.csv",
  "fixtures_2026_v6_feature_engineered.csv"
)

missing_seed_files <- rbindlist(lapply(seq_len(nrow(seed_info)), function(i) {
  data.table(
    seed = seed_info$seed[i],
    file = required_seed_files,
    exists = file.exists(file.path(seed_info$seed_dir[i], required_seed_files))
  )
}))

if (any(!missing_seed_files$exists)) {
  print(missing_seed_files[exists == FALSE])
  stop("Hay seeds incompletas para MC multiseed.")
}

seed_ref_dir <- seed_info$seed_dir[1]
train_v6_file <- file.path(seed_ref_dir, "training_v6_feature_engineered.csv")
fixtures_v6_file <- file.path(seed_ref_dir, "fixtures_2026_v6_feature_engineered.csv")

cat("\n==============================\n")
cat("MONTE CARLO V6 MULTISEED OFFICIAL ANNEX C CONFIG\n")
cat("==============================\n")
cat("N_SIM:", N_SIM, "\n")
cat("Seeds:", paste(seed_info$seed, collapse = ", "), "\n")
cat("GROUP_PROB_SOURCE:", GROUP_PROB_SOURCE, "\n")
cat("KO_PROB_SOURCE:", KO_PROB_SOURCE, "\n")
cat("pred_file:", pred_file, "\n")
cat("mc_out_dir:", mc_out_dir, "\n")

# ------------------------------------------------------------
# 1. Utilidades
# ------------------------------------------------------------

safe_fwrite <- function(x, path) {
  fwrite(x, path)
  invisible(TRUE)
}

normalize_probs <- function(p) {
  p <- as.numeric(p)
  p[!is.finite(p)] <- 0
  p <- pmax(p, 1e-10)
  p / sum(p)
}

rdirichlet1 <- function(alpha) {
  x <- rgamma(length(alpha), shape = alpha, rate = 1)
  x / sum(x)
}

sample_result_3way <- function(p_home, p_draw, p_away, use_dirichlet = FALSE, concentration = 100) {
  p <- normalize_probs(c(p_home, p_draw, p_away))

  if (use_dirichlet) {
    alpha <- pmax(p * concentration, 1e-4)
    p <- rdirichlet1(alpha)
  }

  sample(
    x = c("HomeWin", "Draw", "AwayWin"),
    size = 1,
    prob = p
  )
}

simulate_score_from_result <- function(result) {
  # Marcador compatible con resultado. Solo para desempates de grupo.

  if (result == "Draw") {
    g <- sample(
      x = 0:4,
      size = 1,
      prob = c(0.30, 0.35, 0.22, 0.10, 0.03)
    )
    return(c(gA = g, gB = g))
  }

  loser_goals <- sample(
    x = 0:3,
    size = 1,
    prob = c(0.45, 0.35, 0.15, 0.05)
  )

  margin <- sample(
    x = 1:5,
    size = 1,
    prob = c(0.55, 0.27, 0.12, 0.04, 0.02)
  )

  winner_goals <- loser_goals + margin

  if (result == "HomeWin") return(c(gA = winner_goals, gB = loser_goals))
  if (result == "AwayWin") return(c(gA = loser_goals, gB = winner_goals))

  stop("Resultado no reconocido: ", result)
}


z_from_train <- function(x, mu, sig) {
  if (is.na(sig) || sig == 0) {
    return(rep(0, length(x)))
  }
  (x - mu) / sig
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
first_non_na <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA)
  x[1]
}

mode_non_na <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA)
  names(sort(table(x), decreasing = TRUE))[1]
}
# ------------------------------------------------------------
# 2. Cargar modelos y datos
# ------------------------------------------------------------

load_seed_bundle <- function(seed_dir, seed) {
  metadata_i <- readRDS(file.path(seed_dir, "model_metadata_v6.rds"))

  w <- metadata_i$ensemble_best_weights
  if (is.null(w)) {
    warning("Seed ", seed, ": no encuentro ensemble_best_weights. Uso pesos por defecto.")
    w <- data.table(w_rf = 0.45, w_multinom = 0.15, w_xgb = 0.40)
  } else {
    w <- as.data.table(w)
  }

  list(
    seed = seed,
    seed_dir = seed_dir,
    metadata = metadata_i,
    model_multinom = readRDS(file.path(seed_dir, "model_multinomial_v6.rds")),
    model_rf = readRDS(file.path(seed_dir, "model_random_forest_v6_tuned.rds")),
    model_xgb = xgb.load(file.path(seed_dir, "model_xgboost_v6_tuned.xgb")),
    w_rf = w$w_rf[1],
    w_multinom = w$w_multinom[1],
    w_xgb = w$w_xgb[1]
  )
}

model_bundles <- lapply(seq_len(nrow(seed_info)), function(i) {
  load_seed_bundle(seed_info$seed_dir[i], seed_info$seed[i])
})

metadata <- model_bundles[[1]]$metadata

class_levels <- metadata$result_levels
feature_cols <- metadata$feature_cols
categorical_cols <- metadata$categorical_cols
numeric_cols <- metadata$numeric_cols
xgb_feature_names <- metadata$xgb_feature_names
formula_x_v6 <- metadata$formula_x_v6

if (!all(class_levels == c("HomeWin", "Draw", "AwayWin"))) {
  warning("class_levels no coincide con el orden esperado HomeWin, Draw, AwayWin.")
}

# Verificación conservadora: todos los modelos deben compartir features y clases.
feature_check <- vapply(model_bundles, function(b) identical(b$metadata$feature_cols, feature_cols), logical(1))
class_check <- vapply(model_bundles, function(b) identical(b$metadata$result_levels, class_levels), logical(1))

if (!all(feature_check)) {
  stop("No todas las seeds tienen el mismo feature_cols. No conviene promediar KO sin alinear modelos.")
}
if (!all(class_check)) {
  stop("No todas las seeds tienen los mismos result_levels.")
}

fixtures_pred <- fread(pred_file)
training_v6 <- fread(train_v6_file)
fixtures_v6 <- fread(fixtures_v6_file)

seed_weights <- rbindlist(lapply(model_bundles, function(b) {
  data.table(
    seed = b$seed,
    w_rf = b$w_rf,
    w_multinom = b$w_multinom,
    w_xgb = b$w_xgb
  )
}))

cat("\nPesos ensemble KO por seed:\n")
print(seed_weights)

safe_fwrite(
  seed_weights,
  file.path(mc_out_dir, "mc_multiseed_ensemble_weights_v6.csv")
)

# Levels categóricos desde training de referencia.
cat_levels <- list()
for (cc in categorical_cols) {
  if (cc %in% names(training_v6)) {
    cat_levels[[cc]] <- sort(unique(as.character(training_v6[[cc]])))
  }
}

# ------------------------------------------------------------
# 3. Seleccionar probabilidades de grupos multiseed
# ------------------------------------------------------------

required_base <- c("match_id", "stage", "group", "team_A", "team_B")
missing_base <- setdiff(required_base, names(fixtures_pred))
if (length(missing_base) > 0) {
  stop("Faltan columnas base en fixtures_pred: ", paste(missing_base, collapse = ", "))
}

prob_cols <- c("p_team_A_win", "p_draw", "p_team_B_win")
missing_probs <- setdiff(prob_cols, names(fixtures_pred))
if (length(missing_probs) > 0) {
  stop("Faltan columnas de probabilidad multiseed para grupos: ", paste(missing_probs, collapse = ", "))
}

fixtures_group <- copy(fixtures_pred)

setnames(
  fixtures_group,
  old = prob_cols,
  new = c("p_A_win", "p_draw", "p_B_win")
)

fixtures_group[, p_sum := p_A_win + p_draw + p_B_win]
fixtures_group[, p_A_win := p_A_win / p_sum]
fixtures_group[, p_draw  := p_draw  / p_sum]
fixtures_group[, p_B_win := p_B_win / p_sum]
fixtures_group[, p_sum := NULL]

fixtures_group <- fixtures_group[grepl("Group", stage, ignore.case = TRUE)]
if (nrow(fixtures_group) == 0) {
  stop("No detecto partidos de fase de grupos en fixtures_pred.")
}

teams <- sort(unique(c(fixtures_group$team_A, fixtures_group$team_B)))

groups_dt <- unique(rbind(
  fixtures_group[, .(team = team_A, group)],
  fixtures_group[, .(team = team_B, group)]
))

setorder(groups_dt, group, team)

cat("\nFixtures de grupo:", nrow(fixtures_group), "\n")
cat("Equipos:", length(teams), "\n")
print(fixtures_group[, .N, by = group][order(group)])

# Auditoría de probabilidades de grupo.
group_prob_check <- fixtures_group[, .(
  min_sum = min(p_A_win + p_draw + p_B_win),
  mean_sum = mean(p_A_win + p_draw + p_B_win),
  max_sum = max(p_A_win + p_draw + p_B_win)
)]
cat("\nChequeo suma de probabilidades grupo:\n")
print(group_prob_check)

safe_fwrite(
  fixtures_group,
  file.path(mc_out_dir, "mc_group_fixture_probabilities_multiseed_v6.csv")
)

# ------------------------------------------------------------
# 4. Crear tabla maestra por equipo
# ------------------------------------------------------------

make_team_side_profile <- function(dt, side = c("A", "B")) {

  side <- match.arg(side)
  prefix <- paste0("team_", side, "_")

  team_col <- paste0("team_", side)

  if (!team_col %in% names(dt)) {
    stop("No existe columna: ", team_col)
  }

  # Recuperar Elo individual desde elo_diff y elo_mean.
  tmp <- copy(dt)

  if (!all(c("elo_diff", "elo_mean") %in% names(tmp))) {
    stop("Se requieren elo_diff y elo_mean para reconstruir Elo por equipo.")
  }

  if (side == "A") {
    tmp[, elo_side := elo_mean + elo_diff / 2]
  } else {
    tmp[, elo_side := elo_mean - elo_diff / 2]
  }

  # Columnas prefijadas de ese lado.
  side_cols <- grep(paste0("^", prefix), names(tmp), value = TRUE)

  profile_cols <- c(team_col, side_cols)
  prof <- tmp[, ..profile_cols]

  setnames(prof, team_col, "team")
  setnames(prof, side_cols, sub(paste0("^", prefix), "", side_cols))

  # Añadir variables no prefijadas específicas de equipo.
  prof[, elo := tmp$elo_side]

  rank_col <- paste0("fifa_rank_", side)
  points_col <- paste0("fifa_points_", side)
  confed_col <- paste0("confed_", side)

  if (rank_col %in% names(tmp)) prof[, fifa_rank := tmp[[rank_col]]]
  if (points_col %in% names(tmp)) prof[, fifa_points := tmp[[points_col]]]
  if (confed_col %in% names(tmp)) prof[, confed := tmp[[confed_col]]]

  prof
}

profile_A <- make_team_side_profile(fixtures_v6, "A")
profile_B <- make_team_side_profile(fixtures_v6, "B")

team_profile_long <- rbindlist(list(profile_A, profile_B), fill = TRUE)

numeric_profile_cols <- names(team_profile_long)[sapply(team_profile_long, is.numeric)]
char_profile_cols <- setdiff(names(team_profile_long), c("team", numeric_profile_cols))

team_profile_num <- team_profile_long[
  ,
  lapply(.SD, function(x) mean(x, na.rm = TRUE)),
  by = team,
  .SDcols = numeric_profile_cols
]

team_profile_char <- team_profile_long[
  ,
  lapply(.SD, mode_non_na),
  by = team,
  .SDcols = char_profile_cols
]

team_profile <- merge(team_profile_num, team_profile_char, by = "team", all = TRUE)

# Grupo base
team_profile <- merge(
  team_profile,
  groups_dt,
  by = "team",
  all.x = TRUE
)

# Fallbacks críticos
team_profile[is.na(confed), confed := "UNKNOWN"]
team_profile[is.na(fifa_rank), fifa_rank := median(team_profile$fifa_rank, na.rm = TRUE)]
team_profile[is.na(fifa_points), fifa_points := median(team_profile$fifa_points, na.rm = TRUE)]
team_profile[is.na(elo), elo := median(team_profile$elo, na.rm = TRUE)]

safe_fwrite(
  team_profile,
  file.path(mc_out_dir, "mc_team_profile_v6MS.csv")
)

# ------------------------------------------------------------
# 5. Función para crear features de cualquier cruce
# ------------------------------------------------------------

elo_gap_group_fun <- function(abs_elo_diff) {
  ifelse(abs_elo_diff <= 100, "close",
         ifelse(abs_elo_diff <= 250, "moderate", "large"))
}

elo_level_group_fun <- function(elo_mean) {
  # Ajusta si tu codificación original tenía otros cortes.
  ifelse(elo_mean < 1500, "weak",
         ifelse(elo_mean < 1800, "strong", "elite"))
}

confed_pair_fun <- function(confed_A, confed_B) {
  x <- sort(c(confed_A, confed_B))
  paste0(x[1], "_vs_", x[2])
}

build_match_features <- function(team_A, team_B) {

  pa <- team_profile[team == team_A]
  pb <- team_profile[team == team_B]

  if (nrow(pa) == 0) stop("No encuentro perfil para team_A: ", team_A)
  if (nrow(pb) == 0) stop("No encuentro perfil para team_B: ", team_B)

  row <- data.table(
    match_id = NA_integer_,
    stage = "Knockout MC",
    group = NA_character_,
    team_A = team_A,
    team_B = team_B
  )

  # Variables base de rating
  elo_A <- pa$elo[1]
  elo_B <- pb$elo[1]

  row[, elo_diff := elo_A - elo_B]
  row[, abs_elo_diff := abs(elo_diff)]
  row[, elo_mean := mean(c(elo_A, elo_B), na.rm = TRUE)]
  row[, elo_gap_group := elo_gap_group_fun(abs_elo_diff)]
  row[, elo_level_group := elo_level_group_fun(elo_mean)]

  # Contexto del partido
  row[, neutral := 1]
  row[, host_advantage := as.integer(team_A %in% HOST_TEAMS & !(team_B %in% HOST_TEAMS))]
  row[, tournament_type := "Competitive"]

  # FIFA / confederación
  row[, fifa_rank_A := pa$fifa_rank[1]]
  row[, fifa_rank_B := pb$fifa_rank[1]]
  row[, fifa_points_A := pa$fifa_points[1]]
  row[, fifa_points_B := pb$fifa_points[1]]
  row[, fifa_rank_diff := fifa_rank_A - fifa_rank_B]
  row[, fifa_points_diff := fifa_points_A - fifa_points_B]

  row[, confed_A := pa$confed[1]]
  row[, confed_B := pb$confed[1]]
  row[, same_confed := as.integer(confed_A == confed_B)]
  row[, confed_pair := confed_pair_fun(confed_A, confed_B)]

  # Variables prefijadas por equipo: todo lo que exista en profile y tenga sentido.
  skip_profile_cols <- c("team", "elo", "fifa_rank", "fifa_points", "confed", "group")
  profile_feature_cols <- setdiff(names(team_profile), skip_profile_cols)

  for (cc in profile_feature_cols) {
    row[, paste0("team_A_", cc) := pa[[cc]][1]]
    row[, paste0("team_B_", cc) := pb[[cc]][1]]
  }

  # Diferencias directas para pares conocidos.
  suffixes <- unique(gsub("^team_[AB]_", "", grep("^team_[AB]_", names(row), value = TRUE)))

  for (suf in suffixes) {
    a_col <- paste0("team_A_", suf)
    b_col <- paste0("team_B_", suf)

    if (a_col %in% names(row) && b_col %in% names(row)) {
      if (is.numeric(row[[a_col]]) && is.numeric(row[[b_col]])) {
        diff_col <- paste0(suf, "_diff")
        # Solo si no existe ya.
        if (!diff_col %in% names(row)) {
          row[, (diff_col) := get(a_col) - get(b_col)]
        }
      }
    }
  }

  # Recalcular nombres de diff específicos V5/V6 para evitar errores de sufijo.
  for (w in c(5, 10)) {

    needed <- c(
      paste0("team_A_points_", w), paste0("team_B_points_", w),
      paste0("team_A_gf_", w), paste0("team_B_gf_", w),
      paste0("team_A_ga_", w), paste0("team_B_ga_", w),
      paste0("team_A_gd_", w), paste0("team_B_gd_", w),
      paste0("team_A_win_rate_", w), paste0("team_B_win_rate_", w),
      paste0("team_A_draw_rate_", w), paste0("team_B_draw_rate_", w),
      paste0("team_A_loss_rate_", w), paste0("team_B_loss_rate_", w)
    )

    if (all(needed %in% names(row))) {
      row[, paste0("points_", w, "_diff") := get(paste0("team_A_points_", w)) - get(paste0("team_B_points_", w))]
      row[, paste0("gf_", w, "_diff") := get(paste0("team_A_gf_", w)) - get(paste0("team_B_gf_", w))]
      row[, paste0("ga_", w, "_diff") := get(paste0("team_A_ga_", w)) - get(paste0("team_B_ga_", w))]
      row[, paste0("gd_", w, "_diff") := get(paste0("team_A_gd_", w)) - get(paste0("team_B_gd_", w))]
      row[, paste0("win_rate_", w, "_diff") := get(paste0("team_A_win_rate_", w)) - get(paste0("team_B_win_rate_", w))]
      row[, paste0("draw_rate_", w, "_diff") := get(paste0("team_A_draw_rate_", w)) - get(paste0("team_B_draw_rate_", w))]
      row[, paste0("loss_rate_", w, "_diff") := get(paste0("team_A_loss_rate_", w)) - get(paste0("team_B_loss_rate_", w))]
    }
  }

  # V6 formulas críticas, por si no quedaron creadas desde perfiles.
  for (side in c("team_A", "team_B")) {
    for (w in c(5, 10)) {
      n_col   <- paste0(side, "_n_prev_matches_", w)
      gf_col  <- paste0(side, "_gf_", w)
      ga_col  <- paste0(side, "_ga_", w)
      gd_col  <- paste0(side, "_gd_", w)
      pts_col <- paste0(side, "_points_", w)

      if (all(c(n_col, gf_col, ga_col, gd_col, pts_col) %in% names(row))) {
        row[, paste0(side, "_gf_per_match_", w) := get(gf_col) / pmax(get(n_col), 1)]
        row[, paste0(side, "_ga_per_match_", w) := get(ga_col) / pmax(get(n_col), 1)]
        row[, paste0(side, "_gd_per_match_", w) := get(gd_col) / pmax(get(n_col), 1)]
        row[, paste0(side, "_points_per_match_", w) := get(pts_col) / pmax(get(n_col), 1)]
      }
    }
  }

  for (w in c(5, 10)) {
    if (all(c(
      paste0("team_A_gf_per_match_", w),
      paste0("team_B_gf_per_match_", w),
      paste0("team_A_ga_per_match_", w),
      paste0("team_B_ga_per_match_", w),
      paste0("team_A_gd_per_match_", w),
      paste0("team_B_gd_per_match_", w),
      paste0("team_A_points_per_match_", w),
      paste0("team_B_points_per_match_", w)
    ) %in% names(row))) {
      row[, paste0("gf_per_match_", w, "_diff") := get(paste0("team_A_gf_per_match_", w)) - get(paste0("team_B_gf_per_match_", w))]
      row[, paste0("ga_per_match_", w, "_diff") := get(paste0("team_A_ga_per_match_", w)) - get(paste0("team_B_ga_per_match_", w))]
      row[, paste0("gd_per_match_", w, "_diff") := get(paste0("team_A_gd_per_match_", w)) - get(paste0("team_B_gd_per_match_", w))]
      row[, paste0("points_per_match_", w, "_diff") := get(paste0("team_A_points_per_match_", w)) - get(paste0("team_B_points_per_match_", w))]
    }
  }

  # Momentum
  for (side in c("team_A", "team_B")) {
    if (all(c(
      paste0(side, "_points_per_match_5"), paste0(side, "_points_per_match_10"),
      paste0(side, "_gd_per_match_5"), paste0(side, "_gd_per_match_10"),
      paste0(side, "_gf_per_match_5"), paste0(side, "_gf_per_match_10"),
      paste0(side, "_ga_per_match_5"), paste0(side, "_ga_per_match_10")
    ) %in% names(row))) {
      row[, paste0(side, "_points_momentum_5v10") := get(paste0(side, "_points_per_match_5")) - get(paste0(side, "_points_per_match_10"))]
      row[, paste0(side, "_gd_momentum_5v10") := get(paste0(side, "_gd_per_match_5")) - get(paste0(side, "_gd_per_match_10"))]
      row[, paste0(side, "_gf_momentum_5v10") := get(paste0(side, "_gf_per_match_5")) - get(paste0(side, "_gf_per_match_10"))]
      row[, paste0(side, "_ga_momentum_5v10") := get(paste0(side, "_ga_per_match_5")) - get(paste0(side, "_ga_per_match_10"))]
    }
  }

  if (all(c("team_A_points_momentum_5v10", "team_B_points_momentum_5v10") %in% names(row))) {
    row[, points_momentum_diff := team_A_points_momentum_5v10 - team_B_points_momentum_5v10]
    row[, gd_momentum_diff := team_A_gd_momentum_5v10 - team_B_gd_momentum_5v10]
    row[, gf_momentum_diff := team_A_gf_momentum_5v10 - team_B_gf_momentum_5v10]
    row[, ga_momentum_diff := team_A_ga_momentum_5v10 - team_B_ga_momentum_5v10]
  }

  # Ataque vs defensa
  for (w in c(5, 10)) {
    if (all(c(
      paste0("team_A_gf_per_match_", w),
      paste0("team_B_ga_per_match_", w),
      paste0("team_B_gf_per_match_", w),
      paste0("team_A_ga_per_match_", w)
    ) %in% names(row))) {
      row[, paste0("A_attack_vs_B_def_", w) := get(paste0("team_A_gf_per_match_", w)) - get(paste0("team_B_ga_per_match_", w))]
      row[, paste0("B_attack_vs_A_def_", w) := get(paste0("team_B_gf_per_match_", w)) - get(paste0("team_A_ga_per_match_", w))]
      row[, paste0("attack_matchup_diff_", w) := get(paste0("A_attack_vs_B_def_", w)) - get(paste0("B_attack_vs_A_def_", w))]
    }
  }

  # Elo base
  row[, elo_p_A_base := 1 / (1 + 10^(-elo_diff / 400))]
  row[, elo_p_B_base := 1 - elo_p_A_base]
  row[, elo_favorite_prob := pmax(elo_p_A_base, elo_p_B_base)]
  row[, elo_underdog_prob := pmin(elo_p_A_base, elo_p_B_base)]
  row[, elo_balance := 1 - abs(elo_p_A_base - 0.5) * 2]

  # Draw features
  if (all(c("team_A_draw_rate_5", "team_B_draw_rate_5", "team_A_draw_rate_10", "team_B_draw_rate_10") %in% names(row))) {
    row[, draw_pressure_score :=
          0.35 * (1 - pmin(abs_elo_diff, 500) / 500) +
          0.20 * (1 - pmin(abs(fifa_points_diff), 300) / 300) +
          0.20 * ((team_A_draw_rate_5 + team_B_draw_rate_5) / 2) +
          0.15 * ((team_A_draw_rate_10 + team_B_draw_rate_10) / 2) +
          0.10 * same_confed]
  }

  if (all(c("team_A_gf_per_match_5", "team_B_gf_per_match_5", "team_A_ga_per_match_5", "team_B_ga_per_match_5") %in% names(row))) {
    row[, low_scoring_proxy_5 :=
          -1 * (team_A_gf_per_match_5 + team_B_gf_per_match_5) +
          -1 * (team_A_ga_per_match_5 + team_B_ga_per_match_5)]
    row[, defensive_balance_5 := -abs(team_A_ga_per_match_5 - team_B_ga_per_match_5)]
  }

  if (all(c("team_A_gf_per_match_10", "team_B_gf_per_match_10", "team_A_ga_per_match_10", "team_B_ga_per_match_10") %in% names(row))) {
    row[, low_scoring_proxy_10 :=
          -1 * (team_A_gf_per_match_10 + team_B_gf_per_match_10) +
          -1 * (team_A_ga_per_match_10 + team_B_ga_per_match_10)]
    row[, defensive_balance_10 := -abs(team_A_ga_per_match_10 - team_B_ga_per_match_10)]
  }

  if (all(c("elo_balance", "team_A_draw_rate_10", "team_B_draw_rate_10") %in% names(row))) {
    row[, close_game_x_draw_rate :=
          elo_balance * ((team_A_draw_rate_10 + team_B_draw_rate_10) / 2)]
  }

  # Reliability
  for (side in c("team_A", "team_B")) {
    for (w in c(5, 10)) {
      n_col <- paste0(side, "_n_prev_matches_", w)
      if (n_col %in% names(row)) {
        row[, paste0(side, "_form_reliability_", w) := pmin(get(n_col), 50) / 50]
        row[, paste0(side, "_form_uncertainty_", w) := 1 / sqrt(pmax(get(n_col), 1))]
      }
    }
  }

  for (w in c(5, 10)) {
    if (all(c(
      paste0("team_A_form_reliability_", w),
      paste0("team_B_form_reliability_", w),
      paste0("team_A_form_uncertainty_", w),
      paste0("team_B_form_uncertainty_", w)
    ) %in% names(row))) {
      row[, paste0("form_reliability_", w, "_min") :=
            pmin(get(paste0("team_A_form_reliability_", w)), get(paste0("team_B_form_reliability_", w)))]
      row[, paste0("form_reliability_", w, "_diff") :=
            get(paste0("team_A_form_reliability_", w)) - get(paste0("team_B_form_reliability_", w))]
      row[, paste0("form_uncertainty_", w, "_mean") :=
            (get(paste0("team_A_form_uncertainty_", w)) + get(paste0("team_B_form_uncertainty_", w))) / 2]
    }
  }

  # Interacciones
  if (all(c("elo_diff", "points_per_match_5_diff") %in% names(row))) {
    row[, elo_x_form_5 := elo_diff * points_per_match_5_diff]
  }
  if (all(c("elo_diff", "points_per_match_10_diff") %in% names(row))) {
    row[, elo_x_form_10 := elo_diff * points_per_match_10_diff]
  }
  if (all(c("elo_diff", "attack_matchup_diff_5") %in% names(row))) {
    row[, elo_x_attack_matchup_5 := elo_diff * attack_matchup_diff_5]
  }
  if (all(c("elo_diff", "attack_matchup_diff_10") %in% names(row))) {
    row[, elo_x_attack_matchup_10 := elo_diff * attack_matchup_diff_10]
  }
  if (all(c("elo_balance", "same_confed") %in% names(row))) {
    row[, elo_balance_x_same_confed := elo_balance * same_confed]
  }

  if ("fifa_rank_diff" %in% names(row)) {
    row[, fifa_rank_strength_diff := -fifa_rank_diff]
  }

  # Z-scores según training
  if (!is.null(metadata$scaling_stats)) {
    scaling_stats <- as.data.table(metadata$scaling_stats)
    for (ii in seq_len(nrow(scaling_stats))) {
      v <- scaling_stats$variable[ii]
      if (v %in% names(row)) {
        row[, paste0(v, "_z") := z_from_train(get(v), scaling_stats$mean[ii], scaling_stats$sd[ii])]
      }
    }
  }

  if (all(c("elo_diff_z", "fifa_points_diff_z") %in% names(row))) {
    row[, elo_fifa_disagreement := elo_diff_z - fifa_points_diff_z]
    row[, abs_elo_fifa_disagreement := abs(elo_fifa_disagreement)]
  }

  # Completar columnas faltantes esperadas por el modelo.
  missing_features <- setdiff(feature_cols, names(row))
  if (length(missing_features) > 0) {
    for (cc in missing_features) {
      if (cc %in% names(training_v6)) {
        if (is.numeric(training_v6[[cc]])) {
          row[, (cc) := median(training_v6[[cc]], na.rm = TRUE)]
        } else {
          row[, (cc) := mode_non_na(as.character(training_v6[[cc]]))]
        }
      } else {
        row[, (cc) := 0]
      }
    }
  }

  # Subset compatible
  row_model <- row[, ..feature_cols]

  # Tipos
  for (cc in numeric_cols) {
    if (cc %in% names(row_model)) {
      row_model[, (cc) := as.numeric(get(cc))]
      row_model[!is.finite(get(cc)), (cc) := median(training_v6[[cc]], na.rm = TRUE)]
    }
  }

  for (cc in categorical_cols) {
    if (cc %in% names(row_model)) {
      allowed <- cat_levels[[cc]]
      val <- as.character(row_model[[cc]])
      if (length(allowed) > 0) {
        if (!val %in% allowed) {
          # Fallback a modo de training si aparece un nivel nuevo.
          val <- mode_non_na(as.character(training_v6[[cc]]))
        }
        row_model[, (cc) := factor(val, levels = allowed)]
      } else {
        row_model[, (cc) := as.factor(val)]
      }
    }
  }

  row_model
}

# ------------------------------------------------------------
# 6. Predicción dinámica de un cruce KO
# ------------------------------------------------------------

predict_seed_probs_dynamic <- function(bundle, new_row) {

  # Multinomial
  prob_mult <- predict(
    bundle$model_multinom,
    newdata = new_row,
    type = "probs"
  )

  prob_mult <- as.data.table(as.data.frame(t(as.matrix(prob_mult))))

  if (nrow(prob_mult) != 1 || !all(class_levels %in% names(prob_mult))) {
    prob_mult <- as.data.table(as.list(prob_mult))
  }
  prob_mult <- prob_mult[, class_levels, with = FALSE]

  # RF
  prob_rf <- predict(
    bundle$model_rf,
    data = new_row
  )$predictions

  prob_rf <- as.data.table(prob_rf)
  prob_rf <- prob_rf[, class_levels, with = FALSE]

  # XGB
  formula_i <- bundle$metadata$formula_x_v6
  xgb_names_i <- bundle$metadata$xgb_feature_names

  x_new <- model.matrix(formula_i, data = new_row)[, -1, drop = FALSE]
  x_new <- align_model_matrix(x_new, xgb_names_i)

  raw_xgb <- predict(
    bundle$model_xgb,
    newdata = xgb.DMatrix(x_new)
  )

  prob_xgb <- make_xgb_prob_safe(
    pred_raw = raw_xgb,
    class_levels = class_levels
  )

  # Ensemble específico de esa seed
  prob_ens <- data.table(
    HomeWin = bundle$w_rf * prob_rf$HomeWin + bundle$w_multinom * prob_mult$HomeWin + bundle$w_xgb * prob_xgb$HomeWin,
    Draw    = bundle$w_rf * prob_rf$Draw    + bundle$w_multinom * prob_mult$Draw    + bundle$w_xgb * prob_xgb$Draw,
    AwayWin = bundle$w_rf * prob_rf$AwayWin + bundle$w_multinom * prob_mult$AwayWin + bundle$w_xgb * prob_xgb$AwayWin
  )

  prob_ens[, s := HomeWin + Draw + AwayWin]
  prob_ens[, `:=`(
    HomeWin = HomeWin / s,
    Draw = Draw / s,
    AwayWin = AwayWin / s
  )]
  prob_ens[, s := NULL]

  data.table(
    seed = bundle$seed,
    rf_HomeWin = prob_rf$HomeWin,
    rf_Draw = prob_rf$Draw,
    rf_AwayWin = prob_rf$AwayWin,
    multinom_HomeWin = prob_mult$HomeWin,
    multinom_Draw = prob_mult$Draw,
    multinom_AwayWin = prob_mult$AwayWin,
    xgb_HomeWin = prob_xgb$HomeWin,
    xgb_Draw = prob_xgb$Draw,
    xgb_AwayWin = prob_xgb$AwayWin,
    ensemble_HomeWin = prob_ens$HomeWin,
    ensemble_Draw = prob_ens$Draw,
    ensemble_AwayWin = prob_ens$AwayWin
  )
}

predict_match_probs_dynamic <- function(team_A, team_B) {

  new_row <- build_match_features(team_A, team_B)

  seed_probs <- rbindlist(lapply(model_bundles, function(bundle) {
    predict_seed_probs_dynamic(bundle, new_row)
  }), fill = TRUE)

  out <- seed_probs[, .(
    rf_HomeWin = mean(rf_HomeWin, na.rm = TRUE),
    rf_Draw = mean(rf_Draw, na.rm = TRUE),
    rf_AwayWin = mean(rf_AwayWin, na.rm = TRUE),
    multinom_HomeWin = mean(multinom_HomeWin, na.rm = TRUE),
    multinom_Draw = mean(multinom_Draw, na.rm = TRUE),
    multinom_AwayWin = mean(multinom_AwayWin, na.rm = TRUE),
    xgb_HomeWin = mean(xgb_HomeWin, na.rm = TRUE),
    xgb_Draw = mean(xgb_Draw, na.rm = TRUE),
    xgb_AwayWin = mean(xgb_AwayWin, na.rm = TRUE),
    ensemble_HomeWin = mean(ensemble_HomeWin, na.rm = TRUE),
    ensemble_Draw = mean(ensemble_Draw, na.rm = TRUE),
    ensemble_AwayWin = mean(ensemble_AwayWin, na.rm = TRUE),
    sd_ensemble_HomeWin = sd(ensemble_HomeWin, na.rm = TRUE),
    sd_ensemble_Draw = sd(ensemble_Draw, na.rm = TRUE),
    sd_ensemble_AwayWin = sd(ensemble_AwayWin, na.rm = TRUE)
  )]

  out[, `:=`(
    team_A = team_A,
    team_B = team_B,
    n_seeds = nrow(seed_probs)
  )]

  setcolorder(out, c("team_A", "team_B", "n_seeds"))

  # Para KO usamos el ensemble promedio de las 10 seeds.
  out[, `:=`(
    p_A_win = ensemble_HomeWin,
    p_draw = ensemble_Draw,
    p_B_win = ensemble_AwayWin
  )]

  out[, p_sum := p_A_win + p_draw + p_B_win]
  out[, `:=`(
    p_A_win = p_A_win / p_sum,
    p_draw = p_draw / p_sum,
    p_B_win = p_B_win / p_sum
  )]
  out[, p_sum := NULL]

  out
}
# ------------------------------------------------------------
# 6B. Precalcular probabilidades KO para todos los cruces posibles
# ------------------------------------------------------------

cat("\n==============================\n")
cat("PRECÁLCULO DE PROBABILIDADES KO\n")
cat("==============================\n")

all_ko_pairs <- CJ(
  team_A = teams,
  team_B = teams,
  unique = TRUE
)[team_A != team_B]

cat("Cruces ordenados a precalcular:", nrow(all_ko_pairs), "\n")

pb_ko <- utils::txtProgressBar(
  min = 0,
  max = nrow(all_ko_pairs),
  style = 3
)

t0_ko <- Sys.time()

ko_prob_table <- rbindlist(lapply(seq_len(nrow(all_ko_pairs)), function(i) {
  
  if (i %% 25 == 0 || i == 1 || i == nrow(all_ko_pairs)) {
    utils::setTxtProgressBar(pb_ko, i)
    }
  
  predict_match_probs_dynamic(
    team_A = all_ko_pairs$team_A[i],
    team_B = all_ko_pairs$team_B[i]
  )
}), fill = TRUE)

close(pb_ko)
cat("\nTiempo precálculo KO:\n")
print(Sys.time() - t0_ko)
ko_prob_table[, cache_key := paste(team_A, team_B, sep = "___")]
setkey(ko_prob_table, cache_key)

safe_fwrite(
  ko_prob_table,
  file.path(mc_out_dir, "mc_KO_dynamic_prob_precomputed_all_pairs_v6MS.csv")
)

cat("\nProbabilidades KO precalculadas:", nrow(ko_prob_table), "\n")




# Cache de probabilidades KO para no recalcular el mismo cruce miles de veces.
# ko_prob_cache <- new.env(parent = emptyenv())

get_ko_probs_cached <- function(team_A, team_B) {
  key <- paste(team_A, team_B, sep = "___")
  
  out <- ko_prob_table[J(key)]
  
  if (nrow(out) != 1) {
    stop("No encuentro probabilidad KO precalculada para: ", key)
  }
  
  out
}
simulate_ko_match_dynamic <- function(team_A, team_B) {

  probs <- get_ko_probs_cached(team_A, team_B)

  p <- normalize_probs(c(
    probs$p_A_win,
    probs$p_draw,
    probs$p_B_win
  ))

  if (USE_DIRICHLET_KO) {
    p <- rdirichlet1(pmax(p * DIRICHLET_KO_CONCENTRATION, 1e-4))
  }

  result_90 <- sample(
    x = c("HomeWin", "Draw", "AwayWin"),
    size = 1,
    prob = p
  )

  # Si no hay empate, avanza quien gana en 90.
  if (result_90 == "HomeWin") {
    adv <- team_A
  } else if (result_90 == "AwayWin") {
    adv <- team_B
  } else {
    # Si empatan, decidir penales/prolongación.
    p_pen_A <- p[1] / (p[1] + p[3])
    p_pen_A <- pmin(pmax(p_pen_A, PENALTY_SHRINK_MIN), PENALTY_SHRINK_MAX)
    adv <- ifelse(runif(1) < p_pen_A, team_A, team_B)
  }

  data.table(
    team_A = team_A,
    team_B = team_B,
    p_A_win_90 = probs$p_A_win,
    p_draw_90 = probs$p_draw,
    p_B_win_90 = probs$p_B_win,
    result_90 = result_90,
    winner = adv
  )
}

# ------------------------------------------------------------
# 7. Fase de grupos
# ------------------------------------------------------------

simulate_group_stage <- function(fixtures_group, groups_dt) {

  standings <- copy(groups_dt)
  standings[, `:=`(
    pts = 0L,
    gf = 0L,
    ga = 0L,
    gd = 0L,
    wins = 0L,
    draws = 0L,
    losses = 0L
  )]

  match_results <- vector("list", nrow(fixtures_group))

  for (i in seq_len(nrow(fixtures_group))) {

    m <- fixtures_group[i]

    res <- sample_result_3way(
      p_home = m$p_A_win,
      p_draw = m$p_draw,
      p_away = m$p_B_win,
      use_dirichlet = USE_DIRICHLET_GROUP,
      concentration = DIRICHLET_GROUP_CONCENTRATION
    )

    score <- simulate_score_from_result(res)
    gA <- as.integer(score["gA"])
    gB <- as.integer(score["gB"])

    ptsA <- ifelse(gA > gB, 3L, ifelse(gA == gB, 1L, 0L))
    ptsB <- ifelse(gB > gA, 3L, ifelse(gA == gB, 1L, 0L))

    standings[team == m$team_A, `:=`(
      pts = pts + ptsA,
      gf = gf + gA,
      ga = ga + gB,
      wins = wins + as.integer(gA > gB),
      draws = draws + as.integer(gA == gB),
      losses = losses + as.integer(gA < gB)
    )]

    standings[team == m$team_B, `:=`(
      pts = pts + ptsB,
      gf = gf + gB,
      ga = ga + gA,
      wins = wins + as.integer(gB > gA),
      draws = draws + as.integer(gA == gB),
      losses = losses + as.integer(gB < gA)
    )]

    match_results[[i]] <- data.table(
      match_id = m$match_id,
      group = m$group,
      team_A = m$team_A,
      team_B = m$team_B,
      result = res,
      goals_A = gA,
      goals_B = gB
    )
  }

  standings[, gd := gf - ga]
  standings[, tie_noise := runif(.N)]

  setorder(
    standings,
    group,
    -pts,
    -gd,
    -gf,
    -wins,
    tie_noise
  )

  standings[, group_rank := seq_len(.N), by = group]

  list(
    standings = standings,
    match_results = rbindlist(match_results)
  )
}

get_qualified_32 <- function(standings) {

  top2 <- standings[group_rank <= 2]
  thirds <- standings[group_rank == 3]

  setorder(
    thirds,
    -pts,
    -gd,
    -gf,
    -wins,
    tie_noise
  )

  best_thirds <- thirds[1:8]

  qualified <- rbindlist(list(top2, best_thirds), fill = TRUE)
  qualified[, qualified_type := fifelse(
    group_rank == 1, "group_winner",
    fifelse(group_rank == 2, "group_runner_up", "best_third")
  )]

  qualified
}


# ------------------------------------------------------------
# 8. Tabla oficial FIFA Annex C para mejores terceros
# ------------------------------------------------------------
# Fuente: Regulations for the FIFA World Cup 26, Annexe C.
# La tabla contiene las 495 combinaciones posibles de los 8 terceros clasificados.
# Columnas 1A, 1B, 1D, 1E, 1G, 1I, 1K, 1L indican qué tercer clasificado
# enfrenta a cada ganador de grupo en R32.

annex_c_csv <- '
option,1A,1B,1D,1E,1G,1I,1K,1L,third_groups_key
1,3E,3J,3I,3F,3H,3G,3L,3K,E_F_G_H_I_J_K_L
2,3H,3G,3I,3D,3J,3F,3L,3K,D_F_G_H_I_J_K_L
3,3E,3J,3I,3D,3H,3G,3L,3K,D_E_G_H_I_J_K_L
4,3E,3J,3I,3D,3H,3F,3L,3K,D_E_F_H_I_J_K_L
5,3E,3G,3I,3D,3J,3F,3L,3K,D_E_F_G_I_J_K_L
6,3E,3G,3J,3D,3H,3F,3L,3K,D_E_F_G_H_J_K_L
7,3E,3G,3I,3D,3H,3F,3L,3K,D_E_F_G_H_I_K_L
8,3E,3G,3J,3D,3H,3F,3L,3I,D_E_F_G_H_I_J_L
9,3E,3G,3J,3D,3H,3F,3I,3K,D_E_F_G_H_I_J_K
10,3H,3G,3I,3C,3J,3F,3L,3K,C_F_G_H_I_J_K_L
11,3E,3J,3I,3C,3H,3G,3L,3K,C_E_G_H_I_J_K_L
12,3E,3J,3I,3C,3H,3F,3L,3K,C_E_F_H_I_J_K_L
13,3E,3G,3I,3C,3J,3F,3L,3K,C_E_F_G_I_J_K_L
14,3E,3G,3J,3C,3H,3F,3L,3K,C_E_F_G_H_J_K_L
15,3E,3G,3I,3C,3H,3F,3L,3K,C_E_F_G_H_I_K_L
16,3E,3G,3J,3C,3H,3F,3L,3I,C_E_F_G_H_I_J_L
17,3E,3G,3J,3C,3H,3F,3I,3K,C_E_F_G_H_I_J_K
18,3H,3G,3I,3C,3J,3D,3L,3K,C_D_G_H_I_J_K_L
19,3C,3J,3I,3D,3H,3F,3L,3K,C_D_F_H_I_J_K_L
20,3C,3G,3I,3D,3J,3F,3L,3K,C_D_F_G_I_J_K_L
21,3C,3G,3J,3D,3H,3F,3L,3K,C_D_F_G_H_J_K_L
22,3C,3G,3I,3D,3H,3F,3L,3K,C_D_F_G_H_I_K_L
23,3C,3G,3J,3D,3H,3F,3L,3I,C_D_F_G_H_I_J_L
24,3C,3G,3J,3D,3H,3F,3I,3K,C_D_F_G_H_I_J_K
25,3E,3J,3I,3C,3H,3D,3L,3K,C_D_E_H_I_J_K_L
26,3E,3G,3I,3C,3J,3D,3L,3K,C_D_E_G_I_J_K_L
27,3E,3G,3J,3C,3H,3D,3L,3K,C_D_E_G_H_J_K_L
28,3E,3G,3I,3C,3H,3D,3L,3K,C_D_E_G_H_I_K_L
29,3E,3G,3J,3C,3H,3D,3L,3I,C_D_E_G_H_I_J_L
30,3E,3G,3J,3C,3H,3D,3I,3K,C_D_E_G_H_I_J_K
31,3C,3J,3E,3D,3I,3F,3L,3K,C_D_E_F_I_J_K_L
32,3C,3J,3E,3D,3H,3F,3L,3K,C_D_E_F_H_J_K_L
33,3C,3E,3I,3D,3H,3F,3L,3K,C_D_E_F_H_I_K_L
34,3C,3J,3E,3D,3H,3F,3L,3I,C_D_E_F_H_I_J_L
35,3C,3J,3E,3D,3H,3F,3I,3K,C_D_E_F_H_I_J_K
36,3C,3G,3E,3D,3J,3F,3L,3K,C_D_E_F_G_J_K_L
37,3C,3G,3E,3D,3I,3F,3L,3K,C_D_E_F_G_I_K_L
38,3C,3G,3E,3D,3J,3F,3L,3I,C_D_E_F_G_I_J_L
39,3C,3G,3E,3D,3J,3F,3I,3K,C_D_E_F_G_I_J_K
40,3C,3G,3E,3D,3H,3F,3L,3K,C_D_E_F_G_H_K_L
41,3C,3G,3J,3D,3H,3F,3L,3E,C_D_E_F_G_H_J_L
42,3C,3G,3J,3D,3H,3F,3E,3K,C_D_E_F_G_H_J_K
43,3C,3G,3E,3D,3H,3F,3L,3I,C_D_E_F_G_H_I_L
44,3C,3G,3E,3D,3H,3F,3I,3K,C_D_E_F_G_H_I_K
45,3C,3G,3J,3D,3H,3F,3E,3I,C_D_E_F_G_H_I_J
46,3H,3J,3B,3F,3I,3G,3L,3K,B_F_G_H_I_J_K_L
47,3E,3J,3I,3B,3H,3G,3L,3K,B_E_G_H_I_J_K_L
48,3E,3J,3B,3F,3I,3H,3L,3K,B_E_F_H_I_J_K_L
49,3E,3J,3B,3F,3I,3G,3L,3K,B_E_F_G_I_J_K_L
50,3E,3J,3B,3F,3H,3G,3L,3K,B_E_F_G_H_J_K_L
51,3E,3G,3B,3F,3I,3H,3L,3K,B_E_F_G_H_I_K_L
52,3E,3J,3B,3F,3H,3G,3L,3I,B_E_F_G_H_I_J_L
53,3E,3J,3B,3F,3H,3G,3I,3K,B_E_F_G_H_I_J_K
54,3H,3J,3B,3D,3I,3G,3L,3K,B_D_G_H_I_J_K_L
55,3H,3J,3B,3D,3I,3F,3L,3K,B_D_F_H_I_J_K_L
56,3I,3G,3B,3D,3J,3F,3L,3K,B_D_F_G_I_J_K_L
57,3H,3G,3B,3D,3J,3F,3L,3K,B_D_F_G_H_J_K_L
58,3H,3G,3B,3D,3I,3F,3L,3K,B_D_F_G_H_I_K_L
59,3H,3G,3B,3D,3J,3F,3L,3I,B_D_F_G_H_I_J_L
60,3H,3G,3B,3D,3J,3F,3I,3K,B_D_F_G_H_I_J_K
61,3E,3J,3B,3D,3I,3H,3L,3K,B_D_E_H_I_J_K_L
62,3E,3J,3B,3D,3I,3G,3L,3K,B_D_E_G_I_J_K_L
63,3E,3J,3B,3D,3H,3G,3L,3K,B_D_E_G_H_J_K_L
64,3E,3G,3B,3D,3I,3H,3L,3K,B_D_E_G_H_I_K_L
65,3E,3J,3B,3D,3H,3G,3L,3I,B_D_E_G_H_I_J_L
66,3E,3J,3B,3D,3H,3G,3I,3K,B_D_E_G_H_I_J_K
67,3E,3J,3B,3D,3I,3F,3L,3K,B_D_E_F_I_J_K_L
68,3E,3J,3B,3D,3H,3F,3L,3K,B_D_E_F_H_J_K_L
69,3E,3I,3B,3D,3H,3F,3L,3K,B_D_E_F_H_I_K_L
70,3E,3J,3B,3D,3H,3F,3L,3I,B_D_E_F_H_I_J_L
71,3E,3J,3B,3D,3H,3F,3I,3K,B_D_E_F_H_I_J_K
72,3E,3G,3B,3D,3J,3F,3L,3K,B_D_E_F_G_J_K_L
73,3E,3G,3B,3D,3I,3F,3L,3K,B_D_E_F_G_I_K_L
74,3E,3G,3B,3D,3J,3F,3L,3I,B_D_E_F_G_I_J_L
75,3E,3G,3B,3D,3J,3F,3I,3K,B_D_E_F_G_I_J_K
76,3E,3G,3B,3D,3H,3F,3L,3K,B_D_E_F_G_H_K_L
77,3H,3G,3B,3D,3J,3F,3L,3E,B_D_E_F_G_H_J_L
78,3H,3G,3B,3D,3J,3F,3E,3K,B_D_E_F_G_H_J_K
79,3E,3G,3B,3D,3H,3F,3L,3I,B_D_E_F_G_H_I_L
80,3E,3G,3B,3D,3H,3F,3I,3K,B_D_E_F_G_H_I_K
81,3H,3G,3B,3D,3J,3F,3E,3I,B_D_E_F_G_H_I_J
82,3H,3J,3B,3C,3I,3G,3L,3K,B_C_G_H_I_J_K_L
83,3H,3J,3B,3C,3I,3F,3L,3K,B_C_F_H_I_J_K_L
84,3I,3G,3B,3C,3J,3F,3L,3K,B_C_F_G_I_J_K_L
85,3H,3G,3B,3C,3J,3F,3L,3K,B_C_F_G_H_J_K_L
86,3H,3G,3B,3C,3I,3F,3L,3K,B_C_F_G_H_I_K_L
87,3H,3G,3B,3C,3J,3F,3L,3I,B_C_F_G_H_I_J_L
88,3H,3G,3B,3C,3J,3F,3I,3K,B_C_F_G_H_I_J_K
89,3E,3J,3B,3C,3I,3H,3L,3K,B_C_E_H_I_J_K_L
90,3E,3J,3B,3C,3I,3G,3L,3K,B_C_E_G_I_J_K_L
91,3E,3J,3B,3C,3H,3G,3L,3K,B_C_E_G_H_J_K_L
92,3E,3G,3B,3C,3I,3H,3L,3K,B_C_E_G_H_I_K_L
93,3E,3J,3B,3C,3H,3G,3L,3I,B_C_E_G_H_I_J_L
94,3E,3J,3B,3C,3H,3G,3I,3K,B_C_E_G_H_I_J_K
95,3E,3J,3B,3C,3I,3F,3L,3K,B_C_E_F_I_J_K_L
96,3E,3J,3B,3C,3H,3F,3L,3K,B_C_E_F_H_J_K_L
97,3E,3I,3B,3C,3H,3F,3L,3K,B_C_E_F_H_I_K_L
98,3E,3J,3B,3C,3H,3F,3L,3I,B_C_E_F_H_I_J_L
99,3E,3J,3B,3C,3H,3F,3I,3K,B_C_E_F_H_I_J_K
100,3E,3G,3B,3C,3J,3F,3L,3K,B_C_E_F_G_J_K_L
101,3E,3G,3B,3C,3I,3F,3L,3K,B_C_E_F_G_I_K_L
102,3E,3G,3B,3C,3J,3F,3L,3I,B_C_E_F_G_I_J_L
103,3E,3G,3B,3C,3J,3F,3I,3K,B_C_E_F_G_I_J_K
104,3E,3G,3B,3C,3H,3F,3L,3K,B_C_E_F_G_H_K_L
105,3H,3G,3B,3C,3J,3F,3L,3E,B_C_E_F_G_H_J_L
106,3H,3G,3B,3C,3J,3F,3E,3K,B_C_E_F_G_H_J_K
107,3E,3G,3B,3C,3H,3F,3L,3I,B_C_E_F_G_H_I_L
108,3E,3G,3B,3C,3H,3F,3I,3K,B_C_E_F_G_H_I_K
109,3H,3G,3B,3C,3J,3F,3E,3I,B_C_E_F_G_H_I_J
110,3H,3J,3B,3C,3I,3D,3L,3K,B_C_D_H_I_J_K_L
111,3I,3G,3B,3C,3J,3D,3L,3K,B_C_D_G_I_J_K_L
112,3H,3G,3B,3C,3J,3D,3L,3K,B_C_D_G_H_J_K_L
113,3H,3G,3B,3C,3I,3D,3L,3K,B_C_D_G_H_I_K_L
114,3H,3G,3B,3C,3J,3D,3L,3I,B_C_D_G_H_I_J_L
115,3H,3G,3B,3C,3J,3D,3I,3K,B_C_D_G_H_I_J_K
116,3C,3J,3B,3D,3I,3F,3L,3K,B_C_D_F_I_J_K_L
117,3C,3J,3B,3D,3H,3F,3L,3K,B_C_D_F_H_J_K_L
118,3C,3I,3B,3D,3H,3F,3L,3K,B_C_D_F_H_I_K_L
119,3C,3J,3B,3D,3H,3F,3L,3I,B_C_D_F_H_I_J_L
120,3C,3J,3B,3D,3H,3F,3I,3K,B_C_D_F_H_I_J_K
121,3C,3G,3B,3D,3J,3F,3L,3K,B_C_D_F_G_J_K_L
122,3C,3G,3B,3D,3I,3F,3L,3K,B_C_D_F_G_I_K_L
123,3C,3G,3B,3D,3J,3F,3L,3I,B_C_D_F_G_I_J_L
124,3C,3G,3B,3D,3J,3F,3I,3K,B_C_D_F_G_I_J_K
125,3C,3G,3B,3D,3H,3F,3L,3K,B_C_D_F_G_H_K_L
126,3C,3G,3B,3D,3H,3F,3L,3J,B_C_D_F_G_H_J_L
127,3H,3G,3B,3C,3J,3F,3D,3K,B_C_D_F_G_H_J_K
128,3C,3G,3B,3D,3H,3F,3L,3I,B_C_D_F_G_H_I_L
129,3C,3G,3B,3D,3H,3F,3I,3K,B_C_D_F_G_H_I_K
130,3H,3G,3B,3C,3J,3F,3D,3I,B_C_D_F_G_H_I_J
131,3E,3J,3B,3C,3I,3D,3L,3K,B_C_D_E_I_J_K_L
132,3E,3J,3B,3C,3H,3D,3L,3K,B_C_D_E_H_J_K_L
133,3E,3I,3B,3C,3H,3D,3L,3K,B_C_D_E_H_I_K_L
134,3E,3J,3B,3C,3H,3D,3L,3I,B_C_D_E_H_I_J_L
135,3E,3J,3B,3C,3H,3D,3I,3K,B_C_D_E_H_I_J_K
136,3E,3G,3B,3C,3J,3D,3L,3K,B_C_D_E_G_J_K_L
137,3E,3G,3B,3C,3I,3D,3L,3K,B_C_D_E_G_I_K_L
138,3E,3G,3B,3C,3J,3D,3L,3I,B_C_D_E_G_I_J_L
139,3E,3G,3B,3C,3J,3D,3I,3K,B_C_D_E_G_I_J_K
140,3E,3G,3B,3C,3H,3D,3L,3K,B_C_D_E_G_H_K_L
141,3H,3G,3B,3C,3J,3D,3L,3E,B_C_D_E_G_H_J_L
142,3H,3G,3B,3C,3J,3D,3E,3K,B_C_D_E_G_H_J_K
143,3E,3G,3B,3C,3H,3D,3L,3I,B_C_D_E_G_H_I_L
144,3E,3G,3B,3C,3H,3D,3I,3K,B_C_D_E_G_H_I_K
145,3H,3G,3B,3C,3J,3D,3E,3I,B_C_D_E_G_H_I_J
146,3C,3J,3B,3D,3E,3F,3L,3K,B_C_D_E_F_J_K_L
147,3C,3E,3B,3D,3I,3F,3L,3K,B_C_D_E_F_I_K_L
148,3C,3J,3B,3D,3E,3F,3L,3I,B_C_D_E_F_I_J_L
149,3C,3J,3B,3D,3E,3F,3I,3K,B_C_D_E_F_I_J_K
150,3C,3E,3B,3D,3H,3F,3L,3K,B_C_D_E_F_H_K_L
151,3C,3J,3B,3D,3H,3F,3L,3E,B_C_D_E_F_H_J_L
152,3C,3J,3B,3D,3H,3F,3E,3K,B_C_D_E_F_H_J_K
153,3C,3E,3B,3D,3H,3F,3L,3I,B_C_D_E_F_H_I_L
154,3C,3E,3B,3D,3H,3F,3I,3K,B_C_D_E_F_H_I_K
155,3C,3J,3B,3D,3H,3F,3E,3I,B_C_D_E_F_H_I_J
156,3C,3G,3B,3D,3E,3F,3L,3K,B_C_D_E_F_G_K_L
157,3C,3G,3B,3D,3J,3F,3L,3E,B_C_D_E_F_G_J_L
158,3C,3G,3B,3D,3J,3F,3E,3K,B_C_D_E_F_G_J_K
159,3C,3G,3B,3D,3E,3F,3L,3I,B_C_D_E_F_G_I_L
160,3C,3G,3B,3D,3E,3F,3I,3K,B_C_D_E_F_G_I_K
161,3C,3G,3B,3D,3J,3F,3E,3I,B_C_D_E_F_G_I_J
162,3C,3G,3B,3D,3H,3F,3L,3E,B_C_D_E_F_G_H_L
163,3C,3G,3B,3D,3H,3F,3E,3K,B_C_D_E_F_G_H_K
164,3H,3G,3B,3C,3J,3F,3D,3E,B_C_D_E_F_G_H_J
165,3C,3G,3B,3D,3H,3F,3E,3I,B_C_D_E_F_G_H_I
166,3H,3J,3I,3F,3A,3G,3L,3K,A_F_G_H_I_J_K_L
167,3E,3J,3I,3A,3H,3G,3L,3K,A_E_G_H_I_J_K_L
168,3E,3J,3I,3F,3A,3H,3L,3K,A_E_F_H_I_J_K_L
169,3E,3J,3I,3F,3A,3G,3L,3K,A_E_F_G_I_J_K_L
170,3E,3G,3J,3F,3A,3H,3L,3K,A_E_F_G_H_J_K_L
171,3E,3G,3I,3F,3A,3H,3L,3K,A_E_F_G_H_I_K_L
172,3E,3G,3J,3F,3A,3H,3L,3I,A_E_F_G_H_I_J_L
173,3E,3G,3J,3F,3A,3H,3I,3K,A_E_F_G_H_I_J_K
174,3H,3J,3I,3D,3A,3G,3L,3K,A_D_G_H_I_J_K_L
175,3H,3J,3I,3D,3A,3F,3L,3K,A_D_F_H_I_J_K_L
176,3I,3G,3J,3D,3A,3F,3L,3K,A_D_F_G_I_J_K_L
177,3H,3G,3J,3D,3A,3F,3L,3K,A_D_F_G_H_J_K_L
178,3H,3G,3I,3D,3A,3F,3L,3K,A_D_F_G_H_I_K_L
179,3H,3G,3J,3D,3A,3F,3L,3I,A_D_F_G_H_I_J_L
180,3H,3G,3J,3D,3A,3F,3I,3K,A_D_F_G_H_I_J_K
181,3E,3J,3I,3D,3A,3H,3L,3K,A_D_E_H_I_J_K_L
182,3E,3J,3I,3D,3A,3G,3L,3K,A_D_E_G_I_J_K_L
183,3E,3G,3J,3D,3A,3H,3L,3K,A_D_E_G_H_J_K_L
184,3E,3G,3I,3D,3A,3H,3L,3K,A_D_E_G_H_I_K_L
185,3E,3G,3J,3D,3A,3H,3L,3I,A_D_E_G_H_I_J_L
186,3E,3G,3J,3D,3A,3H,3I,3K,A_D_E_G_H_I_J_K
187,3E,3J,3I,3D,3A,3F,3L,3K,A_D_E_F_I_J_K_L
188,3H,3J,3E,3D,3A,3F,3L,3K,A_D_E_F_H_J_K_L
189,3H,3E,3I,3D,3A,3F,3L,3K,A_D_E_F_H_I_K_L
190,3H,3J,3E,3D,3A,3F,3L,3I,A_D_E_F_H_I_J_L
191,3H,3J,3E,3D,3A,3F,3I,3K,A_D_E_F_H_I_J_K
192,3E,3G,3J,3D,3A,3F,3L,3K,A_D_E_F_G_J_K_L
193,3E,3G,3I,3D,3A,3F,3L,3K,A_D_E_F_G_I_K_L
194,3E,3G,3J,3D,3A,3F,3L,3I,A_D_E_F_G_I_J_L
195,3E,3G,3J,3D,3A,3F,3I,3K,A_D_E_F_G_I_J_K
196,3H,3G,3E,3D,3A,3F,3L,3K,A_D_E_F_G_H_K_L
197,3H,3G,3J,3D,3A,3F,3L,3E,A_D_E_F_G_H_J_L
198,3H,3G,3J,3D,3A,3F,3E,3K,A_D_E_F_G_H_J_K
199,3H,3G,3E,3D,3A,3F,3L,3I,A_D_E_F_G_H_I_L
200,3H,3G,3E,3D,3A,3F,3I,3K,A_D_E_F_G_H_I_K
201,3H,3G,3J,3D,3A,3F,3E,3I,A_D_E_F_G_H_I_J
202,3H,3J,3I,3C,3A,3G,3L,3K,A_C_G_H_I_J_K_L
203,3H,3J,3I,3C,3A,3F,3L,3K,A_C_F_H_I_J_K_L
204,3I,3G,3J,3C,3A,3F,3L,3K,A_C_F_G_I_J_K_L
205,3H,3G,3J,3C,3A,3F,3L,3K,A_C_F_G_H_J_K_L
206,3H,3G,3I,3C,3A,3F,3L,3K,A_C_F_G_H_I_K_L
207,3H,3G,3J,3C,3A,3F,3L,3I,A_C_F_G_H_I_J_L
208,3H,3G,3J,3C,3A,3F,3I,3K,A_C_F_G_H_I_J_K
209,3E,3J,3I,3C,3A,3H,3L,3K,A_C_E_H_I_J_K_L
210,3E,3J,3I,3C,3A,3G,3L,3K,A_C_E_G_I_J_K_L
211,3E,3G,3J,3C,3A,3H,3L,3K,A_C_E_G_H_J_K_L
212,3E,3G,3I,3C,3A,3H,3L,3K,A_C_E_G_H_I_K_L
213,3E,3G,3J,3C,3A,3H,3L,3I,A_C_E_G_H_I_J_L
214,3E,3G,3J,3C,3A,3H,3I,3K,A_C_E_G_H_I_J_K
215,3E,3J,3I,3C,3A,3F,3L,3K,A_C_E_F_I_J_K_L
216,3H,3J,3E,3C,3A,3F,3L,3K,A_C_E_F_H_J_K_L
217,3H,3E,3I,3C,3A,3F,3L,3K,A_C_E_F_H_I_K_L
218,3H,3J,3E,3C,3A,3F,3L,3I,A_C_E_F_H_I_J_L
219,3H,3J,3E,3C,3A,3F,3I,3K,A_C_E_F_H_I_J_K
220,3E,3G,3J,3C,3A,3F,3L,3K,A_C_E_F_G_J_K_L
221,3E,3G,3I,3C,3A,3F,3L,3K,A_C_E_F_G_I_K_L
222,3E,3G,3J,3C,3A,3F,3L,3I,A_C_E_F_G_I_J_L
223,3E,3G,3J,3C,3A,3F,3I,3K,A_C_E_F_G_I_J_K
224,3H,3G,3E,3C,3A,3F,3L,3K,A_C_E_F_G_H_K_L
225,3H,3G,3J,3C,3A,3F,3L,3E,A_C_E_F_G_H_J_L
226,3H,3G,3J,3C,3A,3F,3E,3K,A_C_E_F_G_H_J_K
227,3H,3G,3E,3C,3A,3F,3L,3I,A_C_E_F_G_H_I_L
228,3H,3G,3E,3C,3A,3F,3I,3K,A_C_E_F_G_H_I_K
229,3H,3G,3J,3C,3A,3F,3E,3I,A_C_E_F_G_H_I_J
230,3H,3J,3I,3C,3A,3D,3L,3K,A_C_D_H_I_J_K_L
231,3I,3G,3J,3C,3A,3D,3L,3K,A_C_D_G_I_J_K_L
232,3H,3G,3J,3C,3A,3D,3L,3K,A_C_D_G_H_J_K_L
233,3H,3G,3I,3C,3A,3D,3L,3K,A_C_D_G_H_I_K_L
234,3H,3G,3J,3C,3A,3D,3L,3I,A_C_D_G_H_I_J_L
235,3H,3G,3J,3C,3A,3D,3I,3K,A_C_D_G_H_I_J_K
236,3C,3J,3I,3D,3A,3F,3L,3K,A_C_D_F_I_J_K_L
237,3H,3J,3F,3C,3A,3D,3L,3K,A_C_D_F_H_J_K_L
238,3H,3F,3I,3C,3A,3D,3L,3K,A_C_D_F_H_I_K_L
239,3H,3J,3F,3C,3A,3D,3L,3I,A_C_D_F_H_I_J_L
240,3H,3J,3F,3C,3A,3D,3I,3K,A_C_D_F_H_I_J_K
241,3C,3G,3J,3D,3A,3F,3L,3K,A_C_D_F_G_J_K_L
242,3C,3G,3I,3D,3A,3F,3L,3K,A_C_D_F_G_I_K_L
243,3C,3G,3J,3D,3A,3F,3L,3I,A_C_D_F_G_I_J_L
244,3C,3G,3J,3D,3A,3F,3I,3K,A_C_D_F_G_I_J_K
245,3H,3G,3F,3C,3A,3D,3L,3K,A_C_D_F_G_H_K_L
246,3C,3G,3J,3D,3A,3F,3L,3H,A_C_D_F_G_H_J_L
247,3H,3G,3J,3C,3A,3F,3D,3K,A_C_D_F_G_H_J_K
248,3H,3G,3F,3C,3A,3D,3L,3I,A_C_D_F_G_H_I_L
249,3H,3G,3F,3C,3A,3D,3I,3K,A_C_D_F_G_H_I_K
250,3H,3G,3J,3C,3A,3F,3D,3I,A_C_D_F_G_H_I_J
251,3E,3J,3I,3C,3A,3D,3L,3K,A_C_D_E_I_J_K_L
252,3H,3J,3E,3C,3A,3D,3L,3K,A_C_D_E_H_J_K_L
253,3H,3E,3I,3C,3A,3D,3L,3K,A_C_D_E_H_I_K_L
254,3H,3J,3E,3C,3A,3D,3L,3I,A_C_D_E_H_I_J_L
255,3H,3J,3E,3C,3A,3D,3I,3K,A_C_D_E_H_I_J_K
256,3E,3G,3J,3C,3A,3D,3L,3K,A_C_D_E_G_J_K_L
257,3E,3G,3I,3C,3A,3D,3L,3K,A_C_D_E_G_I_K_L
258,3E,3G,3J,3C,3A,3D,3L,3I,A_C_D_E_G_I_J_L
259,3E,3G,3J,3C,3A,3D,3I,3K,A_C_D_E_G_I_J_K
260,3H,3G,3E,3C,3A,3D,3L,3K,A_C_D_E_G_H_K_L
261,3H,3G,3J,3C,3A,3D,3L,3E,A_C_D_E_G_H_J_L
262,3H,3G,3J,3C,3A,3D,3E,3K,A_C_D_E_G_H_J_K
263,3H,3G,3E,3C,3A,3D,3L,3I,A_C_D_E_G_H_I_L
264,3H,3G,3E,3C,3A,3D,3I,3K,A_C_D_E_G_H_I_K
265,3H,3G,3J,3C,3A,3D,3E,3I,A_C_D_E_G_H_I_J
266,3C,3J,3E,3D,3A,3F,3L,3K,A_C_D_E_F_J_K_L
267,3C,3E,3I,3D,3A,3F,3L,3K,A_C_D_E_F_I_K_L
268,3C,3J,3E,3D,3A,3F,3L,3I,A_C_D_E_F_I_J_L
269,3C,3J,3E,3D,3A,3F,3I,3K,A_C_D_E_F_I_J_K
270,3H,3E,3F,3C,3A,3D,3L,3K,A_C_D_E_F_H_K_L
271,3H,3J,3F,3C,3A,3D,3L,3E,A_C_D_E_F_H_J_L
272,3H,3J,3E,3C,3A,3F,3D,3K,A_C_D_E_F_H_J_K
273,3H,3E,3F,3C,3A,3D,3L,3I,A_C_D_E_F_H_I_L
274,3H,3E,3F,3C,3A,3D,3I,3K,A_C_D_E_F_H_I_K
275,3H,3J,3E,3C,3A,3F,3D,3I,A_C_D_E_F_H_I_J
276,3C,3G,3E,3D,3A,3F,3L,3K,A_C_D_E_F_G_K_L
277,3C,3G,3J,3D,3A,3F,3L,3E,A_C_D_E_F_G_J_L
278,3C,3G,3J,3D,3A,3F,3E,3K,A_C_D_E_F_G_J_K
279,3C,3G,3E,3D,3A,3F,3L,3I,A_C_D_E_F_G_I_L
280,3C,3G,3E,3D,3A,3F,3I,3K,A_C_D_E_F_G_I_K
281,3C,3G,3J,3D,3A,3F,3E,3I,A_C_D_E_F_G_I_J
282,3H,3G,3F,3C,3A,3D,3L,3E,A_C_D_E_F_G_H_L
283,3H,3G,3E,3C,3A,3F,3D,3K,A_C_D_E_F_G_H_K
284,3H,3G,3J,3C,3A,3F,3D,3E,A_C_D_E_F_G_H_J
285,3H,3G,3E,3C,3A,3F,3D,3I,A_C_D_E_F_G_H_I
286,3H,3J,3B,3A,3I,3G,3L,3K,A_B_G_H_I_J_K_L
287,3H,3J,3B,3A,3I,3F,3L,3K,A_B_F_H_I_J_K_L
288,3I,3J,3B,3F,3A,3G,3L,3K,A_B_F_G_I_J_K_L
289,3H,3J,3B,3F,3A,3G,3L,3K,A_B_F_G_H_J_K_L
290,3H,3G,3B,3A,3I,3F,3L,3K,A_B_F_G_H_I_K_L
291,3H,3J,3B,3F,3A,3G,3L,3I,A_B_F_G_H_I_J_L
292,3H,3J,3B,3F,3A,3G,3I,3K,A_B_F_G_H_I_J_K
293,3E,3J,3B,3A,3I,3H,3L,3K,A_B_E_H_I_J_K_L
294,3E,3J,3B,3A,3I,3G,3L,3K,A_B_E_G_I_J_K_L
295,3E,3J,3B,3A,3H,3G,3L,3K,A_B_E_G_H_J_K_L
296,3E,3G,3B,3A,3I,3H,3L,3K,A_B_E_G_H_I_K_L
297,3E,3J,3B,3A,3H,3G,3L,3I,A_B_E_G_H_I_J_L
298,3E,3J,3B,3A,3H,3G,3I,3K,A_B_E_G_H_I_J_K
299,3E,3J,3B,3A,3I,3F,3L,3K,A_B_E_F_I_J_K_L
300,3E,3J,3B,3F,3A,3H,3L,3K,A_B_E_F_H_J_K_L
301,3E,3I,3B,3F,3A,3H,3L,3K,A_B_E_F_H_I_K_L
302,3E,3J,3B,3F,3A,3H,3L,3I,A_B_E_F_H_I_J_L
303,3E,3J,3B,3F,3A,3H,3I,3K,A_B_E_F_H_I_J_K
304,3E,3J,3B,3F,3A,3G,3L,3K,A_B_E_F_G_J_K_L
305,3E,3G,3B,3A,3I,3F,3L,3K,A_B_E_F_G_I_K_L
306,3E,3J,3B,3F,3A,3G,3L,3I,A_B_E_F_G_I_J_L
307,3E,3J,3B,3F,3A,3G,3I,3K,A_B_E_F_G_I_J_K
308,3E,3G,3B,3F,3A,3H,3L,3K,A_B_E_F_G_H_K_L
309,3H,3J,3B,3F,3A,3G,3L,3E,A_B_E_F_G_H_J_L
310,3H,3J,3B,3F,3A,3G,3E,3K,A_B_E_F_G_H_J_K
311,3E,3G,3B,3F,3A,3H,3L,3I,A_B_E_F_G_H_I_L
312,3E,3G,3B,3F,3A,3H,3I,3K,A_B_E_F_G_H_I_K
313,3H,3J,3B,3F,3A,3G,3E,3I,A_B_E_F_G_H_I_J
314,3I,3J,3B,3D,3A,3H,3L,3K,A_B_D_H_I_J_K_L
315,3I,3J,3B,3D,3A,3G,3L,3K,A_B_D_G_I_J_K_L
316,3H,3J,3B,3D,3A,3G,3L,3K,A_B_D_G_H_J_K_L
317,3I,3G,3B,3D,3A,3H,3L,3K,A_B_D_G_H_I_K_L
318,3H,3J,3B,3D,3A,3G,3L,3I,A_B_D_G_H_I_J_L
319,3H,3J,3B,3D,3A,3G,3I,3K,A_B_D_G_H_I_J_K
320,3I,3J,3B,3D,3A,3F,3L,3K,A_B_D_F_I_J_K_L
321,3H,3J,3B,3D,3A,3F,3L,3K,A_B_D_F_H_J_K_L
322,3H,3I,3B,3D,3A,3F,3L,3K,A_B_D_F_H_I_K_L
323,3H,3J,3B,3D,3A,3F,3L,3I,A_B_D_F_H_I_J_L
324,3H,3J,3B,3D,3A,3F,3I,3K,A_B_D_F_H_I_J_K
325,3F,3J,3B,3D,3A,3G,3L,3K,A_B_D_F_G_J_K_L
326,3I,3G,3B,3D,3A,3F,3L,3K,A_B_D_F_G_I_K_L
327,3F,3J,3B,3D,3A,3G,3L,3I,A_B_D_F_G_I_J_L
328,3F,3J,3B,3D,3A,3G,3I,3K,A_B_D_F_G_I_J_K
329,3H,3G,3B,3D,3A,3F,3L,3K,A_B_D_F_G_H_K_L
330,3H,3G,3B,3D,3A,3F,3L,3J,A_B_D_F_G_H_J_L
331,3H,3G,3B,3D,3A,3F,3J,3K,A_B_D_F_G_H_J_K
332,3H,3G,3B,3D,3A,3F,3L,3I,A_B_D_F_G_H_I_L
333,3H,3G,3B,3D,3A,3F,3I,3K,A_B_D_F_G_H_I_K
334,3H,3G,3B,3D,3A,3F,3I,3J,A_B_D_F_G_H_I_J
335,3E,3J,3B,3A,3I,3D,3L,3K,A_B_D_E_I_J_K_L
336,3E,3J,3B,3D,3A,3H,3L,3K,A_B_D_E_H_J_K_L
337,3E,3I,3B,3D,3A,3H,3L,3K,A_B_D_E_H_I_K_L
338,3E,3J,3B,3D,3A,3H,3L,3I,A_B_D_E_H_I_J_L
339,3E,3J,3B,3D,3A,3H,3I,3K,A_B_D_E_H_I_J_K
340,3E,3J,3B,3D,3A,3G,3L,3K,A_B_D_E_G_J_K_L
341,3E,3G,3B,3A,3I,3D,3L,3K,A_B_D_E_G_I_K_L
342,3E,3J,3B,3D,3A,3G,3L,3I,A_B_D_E_G_I_J_L
343,3E,3J,3B,3D,3A,3G,3I,3K,A_B_D_E_G_I_J_K
344,3E,3G,3B,3D,3A,3H,3L,3K,A_B_D_E_G_H_K_L
345,3H,3J,3B,3D,3A,3G,3L,3E,A_B_D_E_G_H_J_L
346,3H,3J,3B,3D,3A,3G,3E,3K,A_B_D_E_G_H_J_K
347,3E,3G,3B,3D,3A,3H,3L,3I,A_B_D_E_G_H_I_L
348,3E,3G,3B,3D,3A,3H,3I,3K,A_B_D_E_G_H_I_K
349,3H,3J,3B,3D,3A,3G,3E,3I,A_B_D_E_G_H_I_J
350,3E,3J,3B,3D,3A,3F,3L,3K,A_B_D_E_F_J_K_L
351,3E,3I,3B,3D,3A,3F,3L,3K,A_B_D_E_F_I_K_L
352,3E,3J,3B,3D,3A,3F,3L,3I,A_B_D_E_F_I_J_L
353,3E,3J,3B,3D,3A,3F,3I,3K,A_B_D_E_F_I_J_K
354,3H,3E,3B,3D,3A,3F,3L,3K,A_B_D_E_F_H_K_L
355,3H,3J,3B,3D,3A,3F,3L,3E,A_B_D_E_F_H_J_L
356,3H,3J,3B,3D,3A,3F,3E,3K,A_B_D_E_F_H_J_K
357,3H,3E,3B,3D,3A,3F,3L,3I,A_B_D_E_F_H_I_L
358,3H,3E,3B,3D,3A,3F,3I,3K,A_B_D_E_F_H_I_K
359,3H,3J,3B,3D,3A,3F,3E,3I,A_B_D_E_F_H_I_J
360,3E,3G,3B,3D,3A,3F,3L,3K,A_B_D_E_F_G_K_L
361,3E,3G,3B,3D,3A,3F,3L,3J,A_B_D_E_F_G_J_L
362,3E,3G,3B,3D,3A,3F,3J,3K,A_B_D_E_F_G_J_K
363,3E,3G,3B,3D,3A,3F,3L,3I,A_B_D_E_F_G_I_L
364,3E,3G,3B,3D,3A,3F,3I,3K,A_B_D_E_F_G_I_K
365,3E,3G,3B,3D,3A,3F,3I,3J,A_B_D_E_F_G_I_J
366,3H,3G,3B,3D,3A,3F,3L,3E,A_B_D_E_F_G_H_L
367,3H,3G,3B,3D,3A,3F,3E,3K,A_B_D_E_F_G_H_K
368,3H,3G,3B,3D,3A,3F,3E,3J,A_B_D_E_F_G_H_J
369,3H,3G,3B,3D,3A,3F,3E,3I,A_B_D_E_F_G_H_I
370,3I,3J,3B,3C,3A,3H,3L,3K,A_B_C_H_I_J_K_L
371,3I,3J,3B,3C,3A,3G,3L,3K,A_B_C_G_I_J_K_L
372,3H,3J,3B,3C,3A,3G,3L,3K,A_B_C_G_H_J_K_L
373,3I,3G,3B,3C,3A,3H,3L,3K,A_B_C_G_H_I_K_L
374,3H,3J,3B,3C,3A,3G,3L,3I,A_B_C_G_H_I_J_L
375,3H,3J,3B,3C,3A,3G,3I,3K,A_B_C_G_H_I_J_K
376,3I,3J,3B,3C,3A,3F,3L,3K,A_B_C_F_I_J_K_L
377,3H,3J,3B,3C,3A,3F,3L,3K,A_B_C_F_H_J_K_L
378,3H,3I,3B,3C,3A,3F,3L,3K,A_B_C_F_H_I_K_L
379,3H,3J,3B,3C,3A,3F,3L,3I,A_B_C_F_H_I_J_L
380,3H,3J,3B,3C,3A,3F,3I,3K,A_B_C_F_H_I_J_K
381,3C,3J,3B,3F,3A,3G,3L,3K,A_B_C_F_G_J_K_L
382,3I,3G,3B,3C,3A,3F,3L,3K,A_B_C_F_G_I_K_L
383,3C,3J,3B,3F,3A,3G,3L,3I,A_B_C_F_G_I_J_L
384,3C,3J,3B,3F,3A,3G,3I,3K,A_B_C_F_G_I_J_K
385,3H,3G,3B,3C,3A,3F,3L,3K,A_B_C_F_G_H_K_L
386,3H,3G,3B,3C,3A,3F,3L,3J,A_B_C_F_G_H_J_L
387,3H,3G,3B,3C,3A,3F,3J,3K,A_B_C_F_G_H_J_K
388,3H,3G,3B,3C,3A,3F,3L,3I,A_B_C_F_G_H_I_L
389,3H,3G,3B,3C,3A,3F,3I,3K,A_B_C_F_G_H_I_K
390,3H,3G,3B,3C,3A,3F,3I,3J,A_B_C_F_G_H_I_J
391,3E,3J,3B,3A,3I,3C,3L,3K,A_B_C_E_I_J_K_L
392,3E,3J,3B,3C,3A,3H,3L,3K,A_B_C_E_H_J_K_L
393,3E,3I,3B,3C,3A,3H,3L,3K,A_B_C_E_H_I_K_L
394,3E,3J,3B,3C,3A,3H,3L,3I,A_B_C_E_H_I_J_L
395,3E,3J,3B,3C,3A,3H,3I,3K,A_B_C_E_H_I_J_K
396,3E,3J,3B,3C,3A,3G,3L,3K,A_B_C_E_G_J_K_L
397,3E,3G,3B,3A,3I,3C,3L,3K,A_B_C_E_G_I_K_L
398,3E,3J,3B,3C,3A,3G,3L,3I,A_B_C_E_G_I_J_L
399,3E,3J,3B,3C,3A,3G,3I,3K,A_B_C_E_G_I_J_K
400,3E,3G,3B,3C,3A,3H,3L,3K,A_B_C_E_G_H_K_L
401,3H,3J,3B,3C,3A,3G,3L,3E,A_B_C_E_G_H_J_L
402,3H,3J,3B,3C,3A,3G,3E,3K,A_B_C_E_G_H_J_K
403,3E,3G,3B,3C,3A,3H,3L,3I,A_B_C_E_G_H_I_L
404,3E,3G,3B,3C,3A,3H,3I,3K,A_B_C_E_G_H_I_K
405,3H,3J,3B,3C,3A,3G,3E,3I,A_B_C_E_G_H_I_J
406,3E,3J,3B,3C,3A,3F,3L,3K,A_B_C_E_F_J_K_L
407,3E,3I,3B,3C,3A,3F,3L,3K,A_B_C_E_F_I_K_L
408,3E,3J,3B,3C,3A,3F,3L,3I,A_B_C_E_F_I_J_L
409,3E,3J,3B,3C,3A,3F,3I,3K,A_B_C_E_F_I_J_K
410,3H,3E,3B,3C,3A,3F,3L,3K,A_B_C_E_F_H_K_L
411,3H,3J,3B,3C,3A,3F,3L,3E,A_B_C_E_F_H_J_L
412,3H,3J,3B,3C,3A,3F,3E,3K,A_B_C_E_F_H_J_K
413,3H,3E,3B,3C,3A,3F,3L,3I,A_B_C_E_F_H_I_L
414,3H,3E,3B,3C,3A,3F,3I,3K,A_B_C_E_F_H_I_K
415,3H,3J,3B,3C,3A,3F,3E,3I,A_B_C_E_F_H_I_J
416,3E,3G,3B,3C,3A,3F,3L,3K,A_B_C_E_F_G_K_L
417,3E,3G,3B,3C,3A,3F,3L,3J,A_B_C_E_F_G_J_L
418,3E,3G,3B,3C,3A,3F,3J,3K,A_B_C_E_F_G_J_K
419,3E,3G,3B,3C,3A,3F,3L,3I,A_B_C_E_F_G_I_L
420,3E,3G,3B,3C,3A,3F,3I,3K,A_B_C_E_F_G_I_K
421,3E,3G,3B,3C,3A,3F,3I,3J,A_B_C_E_F_G_I_J
422,3H,3G,3B,3C,3A,3F,3L,3E,A_B_C_E_F_G_H_L
423,3H,3G,3B,3C,3A,3F,3E,3K,A_B_C_E_F_G_H_K
424,3H,3G,3B,3C,3A,3F,3E,3J,A_B_C_E_F_G_H_J
425,3H,3G,3B,3C,3A,3F,3E,3I,A_B_C_E_F_G_H_I
426,3I,3J,3B,3C,3A,3D,3L,3K,A_B_C_D_I_J_K_L
427,3H,3J,3B,3C,3A,3D,3L,3K,A_B_C_D_H_J_K_L
428,3H,3I,3B,3C,3A,3D,3L,3K,A_B_C_D_H_I_K_L
429,3H,3J,3B,3C,3A,3D,3L,3I,A_B_C_D_H_I_J_L
430,3H,3J,3B,3C,3A,3D,3I,3K,A_B_C_D_H_I_J_K
431,3C,3J,3B,3D,3A,3G,3L,3K,A_B_C_D_G_J_K_L
432,3I,3G,3B,3C,3A,3D,3L,3K,A_B_C_D_G_I_K_L
433,3C,3J,3B,3D,3A,3G,3L,3I,A_B_C_D_G_I_J_L
434,3C,3J,3B,3D,3A,3G,3I,3K,A_B_C_D_G_I_J_K
435,3H,3G,3B,3C,3A,3D,3L,3K,A_B_C_D_G_H_K_L
436,3H,3G,3B,3C,3A,3D,3L,3J,A_B_C_D_G_H_J_L
437,3H,3G,3B,3C,3A,3D,3J,3K,A_B_C_D_G_H_J_K
438,3H,3G,3B,3C,3A,3D,3L,3I,A_B_C_D_G_H_I_L
439,3H,3G,3B,3C,3A,3D,3I,3K,A_B_C_D_G_H_I_K
440,3H,3G,3B,3C,3A,3D,3I,3J,A_B_C_D_G_H_I_J
441,3C,3J,3B,3D,3A,3F,3L,3K,A_B_C_D_F_J_K_L
442,3C,3I,3B,3D,3A,3F,3L,3K,A_B_C_D_F_I_K_L
443,3C,3J,3B,3D,3A,3F,3L,3I,A_B_C_D_F_I_J_L
444,3C,3J,3B,3D,3A,3F,3I,3K,A_B_C_D_F_I_J_K
445,3H,3F,3B,3C,3A,3D,3L,3K,A_B_C_D_F_H_K_L
446,3C,3J,3B,3D,3A,3F,3L,3H,A_B_C_D_F_H_J_L
447,3H,3J,3B,3C,3A,3F,3D,3K,A_B_C_D_F_H_J_K
448,3H,3F,3B,3C,3A,3D,3L,3I,A_B_C_D_F_H_I_L
449,3H,3F,3B,3C,3A,3D,3I,3K,A_B_C_D_F_H_I_K
450,3H,3J,3B,3C,3A,3F,3D,3I,A_B_C_D_F_H_I_J
451,3C,3G,3B,3D,3A,3F,3L,3K,A_B_C_D_F_G_K_L
452,3C,3G,3B,3D,3A,3F,3L,3J,A_B_C_D_F_G_J_L
453,3C,3G,3B,3D,3A,3F,3J,3K,A_B_C_D_F_G_J_K
454,3C,3G,3B,3D,3A,3F,3L,3I,A_B_C_D_F_G_I_L
455,3C,3G,3B,3D,3A,3F,3I,3K,A_B_C_D_F_G_I_K
456,3C,3G,3B,3D,3A,3F,3I,3J,A_B_C_D_F_G_I_J
457,3C,3G,3B,3D,3A,3F,3L,3H,A_B_C_D_F_G_H_L
458,3H,3G,3B,3C,3A,3F,3D,3K,A_B_C_D_F_G_H_K
459,3H,3G,3B,3C,3A,3F,3D,3J,A_B_C_D_F_G_H_J
460,3H,3G,3B,3C,3A,3F,3D,3I,A_B_C_D_F_G_H_I
461,3E,3J,3B,3C,3A,3D,3L,3K,A_B_C_D_E_J_K_L
462,3E,3I,3B,3C,3A,3D,3L,3K,A_B_C_D_E_I_K_L
463,3E,3J,3B,3C,3A,3D,3L,3I,A_B_C_D_E_I_J_L
464,3E,3J,3B,3C,3A,3D,3I,3K,A_B_C_D_E_I_J_K
465,3H,3E,3B,3C,3A,3D,3L,3K,A_B_C_D_E_H_K_L
466,3H,3J,3B,3C,3A,3D,3L,3E,A_B_C_D_E_H_J_L
467,3H,3J,3B,3C,3A,3D,3E,3K,A_B_C_D_E_H_J_K
468,3H,3E,3B,3C,3A,3D,3L,3I,A_B_C_D_E_H_I_L
469,3H,3E,3B,3C,3A,3D,3I,3K,A_B_C_D_E_H_I_K
470,3H,3J,3B,3C,3A,3D,3E,3I,A_B_C_D_E_H_I_J
471,3E,3G,3B,3C,3A,3D,3L,3K,A_B_C_D_E_G_K_L
472,3E,3G,3B,3C,3A,3D,3L,3J,A_B_C_D_E_G_J_L
473,3E,3G,3B,3C,3A,3D,3J,3K,A_B_C_D_E_G_J_K
474,3E,3G,3B,3C,3A,3D,3L,3I,A_B_C_D_E_G_I_L
475,3E,3G,3B,3C,3A,3D,3I,3K,A_B_C_D_E_G_I_K
476,3E,3G,3B,3C,3A,3D,3I,3J,A_B_C_D_E_G_I_J
477,3H,3G,3B,3C,3A,3D,3L,3E,A_B_C_D_E_G_H_L
478,3H,3G,3B,3C,3A,3D,3E,3K,A_B_C_D_E_G_H_K
479,3H,3G,3B,3C,3A,3D,3E,3J,A_B_C_D_E_G_H_J
480,3H,3G,3B,3C,3A,3D,3E,3I,A_B_C_D_E_G_H_I
481,3C,3E,3B,3D,3A,3F,3L,3K,A_B_C_D_E_F_K_L
482,3C,3J,3B,3D,3A,3F,3L,3E,A_B_C_D_E_F_J_L
483,3C,3J,3B,3D,3A,3F,3E,3K,A_B_C_D_E_F_J_K
484,3C,3E,3B,3D,3A,3F,3L,3I,A_B_C_D_E_F_I_L
485,3C,3E,3B,3D,3A,3F,3I,3K,A_B_C_D_E_F_I_K
486,3C,3J,3B,3D,3A,3F,3E,3I,A_B_C_D_E_F_I_J
487,3H,3F,3B,3C,3A,3D,3L,3E,A_B_C_D_E_F_H_L
488,3H,3E,3B,3C,3A,3F,3D,3K,A_B_C_D_E_F_H_K
489,3H,3J,3B,3C,3A,3F,3D,3E,A_B_C_D_E_F_H_J
490,3H,3E,3B,3C,3A,3F,3D,3I,A_B_C_D_E_F_H_I
491,3C,3G,3B,3D,3A,3F,3L,3E,A_B_C_D_E_F_G_L
492,3C,3G,3B,3D,3A,3F,3E,3K,A_B_C_D_E_F_G_K
493,3C,3G,3B,3D,3A,3F,3E,3J,A_B_C_D_E_F_G_J
494,3C,3G,3B,3D,3A,3F,3E,3I,A_B_C_D_E_F_G_I
495,3H,3G,3B,3C,3A,3F,3D,3E,A_B_C_D_E_F_G_H

'

annex_c <- fread(text = annex_c_csv)
annex_c[, option := as.integer(option)]

expected_annex_cols <- c("option", "1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L", "third_groups_key")
missing_annex_cols <- setdiff(expected_annex_cols, names(annex_c))
if (length(missing_annex_cols) > 0) {
  stop("Faltan columnas en annex_c: ", paste(missing_annex_cols, collapse = ", "))
}

if (nrow(annex_c) != 495) {
  stop("Annex C debería tener 495 filas; tiene: ", nrow(annex_c))
}

if (uniqueN(annex_c$third_groups_key) != 495) {
  stop("Annex C debería tener 495 combinaciones únicas de terceros.")
}

safe_fwrite(
  annex_c,
  file.path(mc_out_dir, "annex_c_third_place_table_2026_extracted.csv")
)

normalize_group_letter <- function(x) {
  x <- as.character(x)
  x <- toupper(x)
  x <- gsub("GROUP", "", x)
  x <- gsub("[^A-L]", "", x)
  substr(x, 1, 1)
}

third_key_from_groups <- function(groups) {
  paste(sort(normalize_group_letter(groups)), collapse = "_")
}

# ------------------------------------------------------------
# 9. Bracket oficial FIFA R32 + KO dinámico
# ------------------------------------------------------------

get_team_by_role <- function(qualified, role) {
  role <- as.character(role)
  rank <- as.integer(substr(role, 1, 1))
  grp <- substr(role, 2, 2)

  q <- copy(qualified)
  q[, group_letter := normalize_group_letter(group)]

  hit <- q[group_letter == grp & group_rank == rank]

  if (nrow(hit) != 1) {
    stop(
      "No puedo resolver role ", role,
      ". Filas encontradas: ", nrow(hit),
      ". Grupos terceros/clasificados disponibles: ",
      paste(q[, paste0(group_letter, group_rank)], collapse = ", ")
    )
  }

  hit$team[1]
}

make_official_r32_bracket <- function(qualified) {

  q <- copy(qualified)
  q[, group_letter := normalize_group_letter(group)]

  thirds <- q[qualified_type == "best_third"]

  if (nrow(thirds) != 8) {
    stop("Se esperaban 8 mejores terceros clasificados; encontrados: ", nrow(thirds))
  }

  third_key <- third_key_from_groups(thirds$group_letter)
  annex_row <- annex_c[third_groups_key == third_key]

  if (nrow(annex_row) != 1) {
    stop("No encuentro combinación Annex C para third_key = ", third_key)
  }

  # Match numbers and bracket from FIFA Regulations, Article 12.6.
  # The slots with third-placed teams use Annex C.
  bracket <- data.table(
    slot = 1:16,
    match_no = c("M73", "M74", "M75", "M76", "M77", "M78", "M79", "M80",
                 "M81", "M82", "M83", "M84", "M85", "M86", "M87", "M88"),
    team_A_role = c("2A", "1E", "1F", "1C", "1I", "2E", "1A", "1L",
                    "1D", "1G", "2K", "1H", "1B", "1J", "1K", "2D"),
    team_B_role = c(
      "2B",
      annex_row[["1E"]],
      "2C",
      "2F",
      annex_row[["1I"]],
      "2I",
      annex_row[["1A"]],
      annex_row[["1L"]],
      annex_row[["1D"]],
      annex_row[["1G"]],
      "2L",
      "2J",
      annex_row[["1B"]],
      "2H",
      annex_row[["1K"]],
      "2G"
    )
  )

  bracket[, team_A := vapply(team_A_role, function(r) get_team_by_role(q, r), character(1))]
  bracket[, team_B := vapply(team_B_role, function(r) get_team_by_role(q, r), character(1))]
  bracket[, group_A := substr(team_A_role, 2, 2)]
  bracket[, group_B := substr(team_B_role, 2, 2)]
  bracket[, annex_option := annex_row$option]
  bracket[, third_groups_key := third_key]

  # Safety: teams from the same group must not meet in R32.
  if (any(bracket$group_A == bracket$group_B)) {
    print(bracket[group_A == group_B])
    stop("Bracket inválido: hay cruce de equipos del mismo grupo en R32.")
  }

  bracket
}

simulate_match_row_dynamic <- function(match_no, team_A, team_B, round_name, slot) {
  out <- simulate_ko_match_dynamic(
    team_A = team_A,
    team_B = team_B
  )
  out[, match_no := match_no]
  out[, round := round_name]
  out[, slot := slot]
  out
}

simulate_knockout_dynamic <- function(qualified) {

  bracket32 <- make_official_r32_bracket(qualified)

  # Round of 32: M73-M88
  r32 <- rbindlist(lapply(seq_len(nrow(bracket32)), function(i) {
    simulate_match_row_dynamic(
      match_no = bracket32$match_no[i],
      team_A = bracket32$team_A[i],
      team_B = bracket32$team_B[i],
      round_name = "R32",
      slot = bracket32$slot[i]
    )
  }))

  winner_of <- function(results_dt, match_no_val) {
    x <- results_dt[match_no == match_no_val, winner]
    if (length(x) != 1) stop("No encuentro winner de ", match_no_val)
    x
  }

  # Round of 16 from FIFA Regulations, Article 12.7:
  # M89 W74 v W77; M90 W73 v W75; M91 W76 v W78; M92 W79 v W80;
  # M93 W83 v W84; M94 W81 v W82; M95 W86 v W88; M96 W85 v W87.
  r16_bracket <- data.table(
    slot = 1:8,
    match_no = paste0("M", 89:96),
    from_A = c("M74", "M73", "M76", "M79", "M83", "M81", "M86", "M85"),
    from_B = c("M77", "M75", "M78", "M80", "M84", "M82", "M88", "M87")
  )

  r16_bracket[, team_A := vapply(from_A, function(m) winner_of(r32, m), character(1))]
  r16_bracket[, team_B := vapply(from_B, function(m) winner_of(r32, m), character(1))]

  r16 <- rbindlist(lapply(seq_len(nrow(r16_bracket)), function(i) {
    simulate_match_row_dynamic(
      match_no = r16_bracket$match_no[i],
      team_A = r16_bracket$team_A[i],
      team_B = r16_bracket$team_B[i],
      round_name = "R16",
      slot = r16_bracket$slot[i]
    )
  }))

  # Quarter-finals from Article 12.8:
  # M97 W89 v W90; M98 W93 v W94; M99 W91 v W92; M100 W95 v W96.
  qf_bracket <- data.table(
    slot = 1:4,
    match_no = c("M97", "M98", "M99", "M100"),
    from_A = c("M89", "M93", "M91", "M95"),
    from_B = c("M90", "M94", "M92", "M96")
  )

  qf_bracket[, team_A := vapply(from_A, function(m) winner_of(r16, m), character(1))]
  qf_bracket[, team_B := vapply(from_B, function(m) winner_of(r16, m), character(1))]

  qf <- rbindlist(lapply(seq_len(nrow(qf_bracket)), function(i) {
    simulate_match_row_dynamic(
      match_no = qf_bracket$match_no[i],
      team_A = qf_bracket$team_A[i],
      team_B = qf_bracket$team_B[i],
      round_name = "QF",
      slot = qf_bracket$slot[i]
    )
  }))

  # Semi-finals from Article 12.9:
  # M101 W97 v W98; M102 W99 v W100.
  sf_bracket <- data.table(
    slot = 1:2,
    match_no = c("M101", "M102"),
    from_A = c("M97", "M99"),
    from_B = c("M98", "M100")
  )

  sf_bracket[, team_A := vapply(from_A, function(m) winner_of(qf, m), character(1))]
  sf_bracket[, team_B := vapply(from_B, function(m) winner_of(qf, m), character(1))]

  sf <- rbindlist(lapply(seq_len(nrow(sf_bracket)), function(i) {
    simulate_match_row_dynamic(
      match_no = sf_bracket$match_no[i],
      team_A = sf_bracket$team_A[i],
      team_B = sf_bracket$team_B[i],
      round_name = "SF",
      slot = sf_bracket$slot[i]
    )
  }))

  # Final from Article 12.11: M104 W101 v W102.
  final <- simulate_match_row_dynamic(
    match_no = "M104",
    team_A = winner_of(sf, "M101"),
    team_B = winner_of(sf, "M102"),
    round_name = "Final",
    slot = 1L
  )

  list(
    bracket32 = bracket32,
    r16_bracket = r16_bracket,
    qf_bracket = qf_bracket,
    sf_bracket = sf_bracket,
    r32 = r32,
    r16 = r16,
    qf = qf,
    sf = sf,
    final = final,
    champion = final$winner
  )
}
# ------------------------------------------------------------
# 10. Loop Monte Carlo
# ------------------------------------------------------------

stage_counts <- data.table(
  team = teams,
  qualified_R32 = 0L,
  reached_R16 = 0L,
  reached_QF = 0L,
  reached_SF = 0L,
  reached_Final = 0L,
  champion = 0L
)

position_counts <- data.table(
  team = teams,
  group = groups_dt[match(teams, team), group],
  pos1 = 0L,
  pos2 = 0L,
  pos3 = 0L,
  pos4 = 0L,
  qualified_R32 = 0L,
  points_sum = 0,
  gd_sum = 0,
  gf_sum = 0
)

ko_match_log_sample <- list()
champions <- character(N_SIM)

cat("\n==============================\n")
cat("INICIANDO SIMULACIONES V6MS\n")
cat("==============================\n")

pb <- utils::txtProgressBar(
  min = 0,
  max = N_SIM,
  style = 3
)

t0 <- Sys.time()

for (sim in seq_len(N_SIM)) {
  
  if (sim %% 50 == 0 || sim == 1 || sim == N_SIM) {
    utils::setTxtProgressBar(pb, sim)
  }
  gs <- simulate_group_stage(fixtures_group, groups_dt)
  standings <- gs$standings

  # Posiciones de grupo
  for (r in 1:4) {
    teams_r <- standings[group_rank == r, team]
    col_r <- paste0("pos", r)
    position_counts[team %in% teams_r, (col_r) := get(col_r) + 1L]
  }

  position_counts[standings, on = "team", `:=`(
    points_sum = points_sum + i.pts,
    gd_sum = gd_sum + i.gd,
    gf_sum = gf_sum + i.gf
  )]

  qualified <- get_qualified_32(standings)
  q_teams <- qualified$team

  stage_counts[team %in% q_teams, qualified_R32 := qualified_R32 + 1L]
  position_counts[team %in% q_teams, qualified_R32 := qualified_R32 + 1L]

  ko <- simulate_knockout_dynamic(qualified)

  r16_teams <- ko$r32$winner
  qf_teams  <- ko$r16$winner
  sf_teams  <- ko$qf$winner
  f_teams   <- ko$sf$winner
  champ     <- ko$champion

  stage_counts[team %in% r16_teams, reached_R16 := reached_R16 + 1L]
  stage_counts[team %in% qf_teams,  reached_QF := reached_QF + 1L]
  stage_counts[team %in% sf_teams,  reached_SF := reached_SF + 1L]
  stage_counts[team %in% f_teams,   reached_Final := reached_Final + 1L]
  stage_counts[team == champ, champion := champion + 1L]

  champions[sim] <- champ

  # Guardar log pequeño de KO de las primeras simulaciones para auditoría.
  if (sim <= 5) {
    ko_match_log_sample[[sim]] <- rbindlist(list(
      ko$r32, ko$r16, ko$qf, ko$sf, ko$final
    ), fill = TRUE)[, sim_id := sim][]
  }
}
close(pb)

cat("\nTiempo total simulaciones:\n")
print(Sys.time() - t0)
# ------------------------------------------------------------
# 11. Resúmenes
# ------------------------------------------------------------

stage_probs <- copy(stage_counts)

prob_cols <- setdiff(names(stage_probs), "team")
for (cc in prob_cols) {
  stage_probs[, (cc) := get(cc) / N_SIM]
}

setorder(stage_probs, -champion, -reached_Final, -reached_SF)

safe_fwrite(
  stage_probs,
  file.path(mc_out_dir, "mc_team_stage_probabilities_v6MS_official_annexC.csv")
)

position_probs <- copy(position_counts)

position_probs[, `:=`(
  p_pos1 = pos1 / N_SIM,
  p_pos2 = pos2 / N_SIM,
  p_pos3 = pos3 / N_SIM,
  p_pos4 = pos4 / N_SIM,
  p_qualified_R32 = qualified_R32 / N_SIM,
  avg_points = points_sum / N_SIM,
  avg_gd = gd_sum / N_SIM,
  avg_gf = gf_sum / N_SIM
)]

setorder(position_probs, group, -p_pos1, -p_pos2, -p_qualified_R32)

safe_fwrite(
  position_probs,
  file.path(mc_out_dir, "mc_group_position_probabilities_v6MS_official_annexC.csv")
)

champion_probs <- data.table(champion = champions)[
  ,
  .N,
  by = champion
][
  order(-N)
]

champion_probs[, p_champion := N / N_SIM]

safe_fwrite(
  champion_probs,
  file.path(mc_out_dir, "mc_simulation_champions_v6MS_official_annexC.csv")
)

group_summary <- position_probs[
  ,
  .(
    top_group_winner = team[which.max(p_pos1)],
    max_p_pos1 = max(p_pos1),
    top_qualified = team[which.max(p_qualified_R32)],
    max_p_qualified = max(p_qualified_R32)
  ),
  by = group
][order(group)]

safe_fwrite(
  group_summary,
  file.path(mc_out_dir, "mc_group_summary_v6MS_official_annexC.csv")
)

if (length(ko_match_log_sample) > 0) {
  ko_log_dt <- rbindlist(ko_match_log_sample, fill = TRUE)
  safe_fwrite(
    ko_log_dt,
    file.path(mc_out_dir, "mc_KO_match_log_sample_first5_v6MS.csv")
  )
}

# Exportar cache de probabilidades KO calculadas
# cache_keys <- ls(envir = ko_prob_cache)
# if (length(cache_keys) > 0) {
#   ko_cache_dt <- rbindlist(lapply(cache_keys, function(k) {
#     x <- get(k, envir = ko_prob_cache)
#     x[, cache_key := k]
#     x
#   }), fill = TRUE)
# 
#   safe_fwrite(
#     ko_cache_dt,
#     file.path(mc_out_dir, "mc_KO_dynamic_prob_cache_v6MS.csv")
#   )
# }

metadata_out <- list(
  n_sim = N_SIM,
  group_prob_source = GROUP_PROB_SOURCE,
  ko_prob_source = KO_PROB_SOURCE,
  use_dirichlet_group = USE_DIRICHLET_GROUP,
  dirichlet_group_concentration = DIRICHLET_GROUP_CONCENTRATION,
  use_dirichlet_ko = USE_DIRICHLET_KO,
  dirichlet_ko_concentration = DIRICHLET_KO_CONCENTRATION,
  penalty_shrink_min = PENALTY_SHRINK_MIN,
  penalty_shrink_max = PENALTY_SHRINK_MAX,
  host_teams = HOST_TEAMS,
  seed_weights = seed_weights,
  n_seeds = nrow(seed_info),
  seed_values = seed_info$seed,
  group_probability_file = pred_file,
  note = "V6 multiseed: group stage uses mean probabilities from 10 seeds; knockout builds dynamic match features, predicts each KO pair with all seed models, averages ensemble probabilities, and uses official FIFA Annex C table for the 495 third-place combinations and official knockout bracket pairings from Articles 12.6-12.11."
)

saveRDS(
  metadata_out,
  file.path(mc_out_dir, "mc_simulation_metadata_v6MS_official_annexC.rds")
)

# ------------------------------------------------------------
# 12. Reporte consola
# ------------------------------------------------------------

cat("\n==============================\n")
cat("MONTE CARLO V6MS COMPLETADO\n")
cat("==============================\n")

cat("\nTop 20 campeón:\n")
print(champion_probs[1:min(20, .N)])

cat("\nTop 20 llegar a final:\n")
print(stage_probs[order(-reached_Final)][1:min(20, .N), .(
  team,
  qualified_R32,
  reached_R16,
  reached_QF,
  reached_SF,
  reached_Final,
  champion
)])

cat("\nResumen por grupo:\n")
print(group_summary)

cat("\nCruces KO precalculados:", nrow(ko_prob_table), "\n")

cat("\nArchivos generados en:\n")
cat(mc_out_dir, "\n")
