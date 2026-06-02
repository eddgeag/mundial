# ============================================================
# 07_golden_boot_v6MS.R
# Predicción de máximo goleador Mundial 2026
# usando goalscorers.csv + Monte Carlo V6MS
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
})

# ------------------------------------------------------------
# 0. Configuración
# ------------------------------------------------------------

data_dir <- "datos"

mc_dir <- file.path(
  "resultados_server",
  "v6_multiseed_consolidated",
  "MONTE_CARLO_V6_MULTISEED_OFFICIAL_ANNEXC"
)

out_dir <- file.path(
  mc_dir,
  "GOLDEN_BOOT_V6MS"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

goalscorers_file <- file.path(data_dir, "goalscorers.csv")

team_stage_file <- file.path(
  mc_dir,
  "mc_team_stage_probabilities_v6MS_official_annexC.csv"
)

set.seed(20260602)

N_SIM <- 100000

# Desde qué año usar goles recientes de selección
RECENT_FROM <- "2022-01-01"

# Mínimo de goles recientes para entrar automáticamente como candidato
MIN_RECENT_GOALS <- 3

# Número máximo de candidatos por selección
TOP_N_PER_TEAM <- 4

# Modelo de conteo:
# "poisson" o "negbin"
GOAL_MODEL <- "negbin"

# Sobredispersión para binomial negativa.
# Mayor size = más parecido a Poisson.
NB_SIZE <-10

# ------------------------------------------------------------
# 1. Cargar archivos
# ------------------------------------------------------------

if (!file.exists(goalscorers_file)) {
  stop("No existe: ", goalscorers_file)
}

if (!file.exists(team_stage_file)) {
  stop("No existe: ", team_stage_file)
}

goals <- fread(goalscorers_file)
team_stage <- fread(team_stage_file)

cat("\n==============================\n")
cat("GOLDEN BOOT V6MS\n")
cat("==============================\n")
cat("goalscorers_file:", goalscorers_file, "\n")
cat("team_stage_file:", team_stage_file, "\n")
cat("N_SIM:", N_SIM, "\n")
cat("RECENT_FROM:", RECENT_FROM, "\n")
cat("GOAL_MODEL:", GOAL_MODEL, "\n")

# ------------------------------------------------------------
# 2. Auditoría columnas
# ------------------------------------------------------------

required_goals_cols <- c(
  "date", "home_team", "away_team", "team",
  "scorer", "minute", "own_goal", "penalty"
)

missing_goals_cols <- setdiff(required_goals_cols, names(goals))

if (length(missing_goals_cols) > 0) {
  stop(
    "Faltan columnas en goalscorers.csv: ",
    paste(missing_goals_cols, collapse = ", ")
  )
}

required_stage_cols <- c(
  "team", "qualified_R32", "reached_R16", "reached_QF",
  "reached_SF", "reached_Final", "champion"
)

missing_stage_cols <- setdiff(required_stage_cols, names(team_stage))

if (length(missing_stage_cols) > 0) {
  stop(
    "Faltan columnas en team_stage: ",
    paste(missing_stage_cols, collapse = ", ")
  )
}

# ------------------------------------------------------------
# 3. Limpieza básica
# ------------------------------------------------------------

goals[, date := as.IDate(date)]

# Normalizar lógicos por si vienen como texto
to_logical_safe <- function(x) {
  if (is.logical(x)) return(x)
  x <- toupper(as.character(x))
  fifelse(x %in% c("TRUE", "T", "1", "YES", "SI", "SÍ"), TRUE,
          fifelse(x %in% c("FALSE", "F", "0", "NO"), FALSE, NA))
}

goals[, own_goal := to_logical_safe(own_goal)]
goals[, penalty := to_logical_safe(penalty)]

# Quitar filas sin anotador
goals <- goals[!is.na(scorer) & scorer != ""]

# Quitar autogoles
goals_real <- goals[is.na(own_goal) | own_goal == FALSE]

# ------------------------------------------------------------
# 4. Filtrar selecciones del Mundial según tu Monte Carlo
# ------------------------------------------------------------

teams_2026 <- unique(team_stage$team)

goals_wc_teams <- goals_real[team %in% teams_2026]

cat("\nSelecciones en Monte Carlo:", length(teams_2026), "\n")
cat("Goles históricos de equipos 2026:", nrow(goals_wc_teams), "\n")

# ------------------------------------------------------------
# 5. Goles recientes por jugador
# ------------------------------------------------------------

goals_recent <- goals_wc_teams[date >= as.IDate(RECENT_FROM)]

scorers_recent <- goals_recent[
  ,
  .(
    recent_goals = .N,
    recent_penalty_goals = sum(penalty == TRUE, na.rm = TRUE),
    first_recent_goal = min(date, na.rm = TRUE),
    last_recent_goal = max(date, na.rm = TRUE)
  ),
  by = .(team, scorer)
]

# Goles totales históricos como desempate / soporte
scorers_all <- goals_wc_teams[
  ,
  .(
    total_intl_goals_in_file = .N,
    total_penalty_goals_in_file = sum(penalty == TRUE, na.rm = TRUE)
  ),
  by = .(team, scorer)
]

scorers <- merge(
  scorers_recent,
  scorers_all,
  by = c("team", "scorer"),
  all.x = TRUE
)

# ------------------------------------------------------------
# 6. Quedarse con candidatos razonables
# ------------------------------------------------------------

setorder(scorers, team, -recent_goals, -total_intl_goals_in_file)

candidates <- scorers[
  recent_goals >= MIN_RECENT_GOALS,
  head(.SD, TOP_N_PER_TEAM),
  by = team
]

cat("\nCandidatos automáticos:", nrow(candidates), "\n")

# ------------------------------------------------------------
# 7. Partidos esperados por selección
# ------------------------------------------------------------
# Aproximación:
# Todos juegan 3 partidos de grupo.
# Luego sumamos probabilidades de alcanzar rondas sucesivas.
#
# En tu archivo las columnas son:
# qualified_R32, reached_R16, reached_QF, reached_SF, reached_Final, champion
#
# Para partidos esperados:
# 3 + P(R32) + P(R16) + P(QF) + P(SF) + P(Final)
#
# El campeón no suma partido adicional porque la final ya está incluida.

team_stage[, expected_matches :=
             3 +
             qualified_R32 +
             reached_R16 +
             reached_QF +
             reached_SF +
             reached_Final
]

team_stage[, final_or_champion_prob := reached_Final]
team_stage[, deep_run_score := reached_QF + reached_SF + reached_Final + champion]

# ------------------------------------------------------------
# 8. Unir candidatos con Monte Carlo
# ------------------------------------------------------------

gb <- merge(
  candidates,
  team_stage,
  by = "team",
  all.x = TRUE
)

if (any(is.na(gb$expected_matches))) {
  warning("Hay candidatos sin expected_matches. Revisar nombres de selecciones.")
}

# ------------------------------------------------------------
# 9. Heurísticas de tasa goleadora
# ------------------------------------------------------------
# Como goalscorers.csv no tiene minutos, usamos goles recientes con selección.
# Convertimos goles recientes a una tasa proxy por torneo.
#
# recent_goal_strength:
# - goles recientes
# - penal cuenta, pero se penaliza ligeramente para no inflar demasiado
# - goles históricos ayudan como soporte menor

gb[, non_penalty_recent_goals := recent_goals - recent_penalty_goals]

gb[, recent_goal_strength :=
     non_penalty_recent_goals +
     0.75 * recent_penalty_goals +
     0.10 * pmin(total_intl_goals_in_file, 30)
]

# Escalamiento interno.
# Esto evita lambdas absurdamente altas.
gb[, recent_goal_strength_scaled :=
     recent_goal_strength / max(recent_goal_strength, na.rm = TRUE)
]

# Penalero probable:
# Si más del 25% de sus goles recientes fueron de penal, lo marcamos como candidato a penalero.
gb[, penalty_taker_proxy :=
     fifelse(recent_goals > 0 & recent_penalty_goals / recent_goals >= 0.25, 1, 0)
]

# Bonus moderado por penales
gb[, penalty_bonus := fifelse(penalty_taker_proxy == 1, 1.15, 1.00)]

# Ajuste por carrera larga del equipo
gb[, team_run_bonus :=
     1 +
     0.30 * reached_QF +
     0.25 * reached_SF +
     0.20 * reached_Final +
     0.15 * champion
]

# ------------------------------------------------------------
# 10. Lambda de goles esperados en el torneo
# ------------------------------------------------------------
# Esta lambda es una aproximación.
# Para calibrarla, queremos que los favoritos queden usualmente ~3-6 goles esperados
# y que el máximo simulado caiga frecuentemente en 5-8 goles.

gb[, lambda_goals :=
     expected_matches *
     (0.12 + 0.50 * recent_goal_strength_scaled) *
     penalty_bonus *
     team_run_bonus
]
# Evitar extremos
gb[, lambda_goals := pmax(lambda_goals, 0.05)]
gb[, lambda_goals := pmin(lambda_goals, 5.5)]
setorder(gb, -lambda_goals)

cat("\nTop candidatos por lambda:\n")
print(gb[1:min(.N, 20), .(
  team, scorer, recent_goals, recent_penalty_goals,
  expected_matches, lambda_goals
)])

# ------------------------------------------------------------
# 11. Simulación de goles por jugador
# ------------------------------------------------------------

player_ids <- gb[, paste(team, scorer, sep = " | ")]
lambda <- gb$lambda_goals

n_players <- length(player_ids)

cat("\nSimulando goles...\n")
cat("Jugadores:", n_players, "\n")
cat("Simulaciones:", N_SIM, "\n")

# Matriz simulada: filas = simulaciones, columnas = jugadores
if (GOAL_MODEL == "poisson") {
  sim_mat <- matrix(
    rpois(N_SIM * n_players, lambda = rep(lambda, each = N_SIM)),
    nrow = N_SIM,
    ncol = n_players,
    byrow = FALSE
  )
} else if (GOAL_MODEL == "negbin") {
  sim_mat <- matrix(
    rnbinom(
      N_SIM * n_players,
      size = NB_SIZE,
      mu = rep(lambda, each = N_SIM)
    ),
    nrow = N_SIM,
    ncol = n_players,
    byrow = FALSE
  )
} else {
  stop("GOAL_MODEL debe ser 'poisson' o 'negbin'")
}

# ------------------------------------------------------------
# 12. Ganador Bota de Oro por simulación
# ------------------------------------------------------------
# Si hay empate, se elige aleatoriamente entre empatados.
# En la vida real hay criterios de desempate, pero para quiniela esto es suficiente.

winner_idx <- integer(N_SIM)
max_goals <- integer(N_SIM)

for (i in seq_len(N_SIM)) {
  row_goals <- sim_mat[i, ]
  mg <- max(row_goals)
  tied <- which(row_goals == mg)
  winner_idx[i] <- sample(tied, 1)
  max_goals[i] <- mg
}

winner_tab <- as.data.table(table(winner_idx))
winner_tab[, winner_idx := as.integer(as.character(winner_idx))]
setnames(winner_tab, "N", "golden_boot_wins")

winner_tab[, golden_boot_prob := golden_boot_wins / N_SIM]

gb_results <- copy(gb)
gb_results[, winner_idx := .I]

gb_results <- merge(
  gb_results,
  winner_tab,
  by = "winner_idx",
  all.x = TRUE
)

gb_results[is.na(golden_boot_wins), golden_boot_wins := 0]
gb_results[is.na(golden_boot_prob), golden_boot_prob := 0]

# Goles simulados promedio por jugador
mean_goals <- colMeans(sim_mat)
p_ge_4 <- colMeans(sim_mat >= 4)
p_ge_5 <- colMeans(sim_mat >= 5)
p_ge_6 <- colMeans(sim_mat >= 6)
p_ge_7 <- colMeans(sim_mat >= 7)

gb_results[, sim_mean_goals := mean_goals[winner_idx]]
gb_results[, prob_4plus_goals := p_ge_4[winner_idx]]
gb_results[, prob_5plus_goals := p_ge_5[winner_idx]]
gb_results[, prob_6plus_goals := p_ge_6[winner_idx]]
gb_results[, prob_7plus_goals := p_ge_7[winner_idx]]

setorder(gb_results, -golden_boot_prob, -lambda_goals)

# ------------------------------------------------------------
# 13. Distribución del número máximo de goles
# ------------------------------------------------------------

max_goals_dist <- as.data.table(table(max_goals))
setnames(max_goals_dist, c("max_goals", "n_sim"))
max_goals_dist[, max_goals := as.integer(as.character(max_goals))]
max_goals_dist[, probability := n_sim / N_SIM]
setorder(max_goals_dist, max_goals)

# Valor recomendado para quiniela: moda del máximo de goles
recommended_goals <- max_goals_dist[which.max(probability), max_goals]

# ------------------------------------------------------------
# 14. Guardar resultados
# ------------------------------------------------------------

fwrite(
  gb_results,
  file.path(out_dir, "golden_boot_player_probabilities_v6MS.csv")
)

fwrite(
  gb_results[1:min(.N, 30)],
  file.path(out_dir, "golden_boot_top30_v6MS.csv")
)

fwrite(
  max_goals_dist,
  file.path(out_dir, "golden_boot_max_goals_distribution_v6MS.csv")
)

metadata <- data.table(
  parameter = c(
    "N_SIM",
    "RECENT_FROM",
    "MIN_RECENT_GOALS",
    "TOP_N_PER_TEAM",
    "GOAL_MODEL",
    "NB_SIZE",
    "recommended_goals"
  ),
  value = c(
    N_SIM,
    RECENT_FROM,
    MIN_RECENT_GOALS,
    TOP_N_PER_TEAM,
    GOAL_MODEL,
    NB_SIZE,
    recommended_goals
  )
)

fwrite(
  metadata,
  file.path(out_dir, "golden_boot_metadata_v6MS.csv")
)

# ------------------------------------------------------------
# 15. Resumen en consola
# ------------------------------------------------------------

cat("\n==============================\n")
cat("TOP 20 BOTA DE ORO\n")
cat("==============================\n")

print(
  gb_results[1:min(.N, 20), .(
    rank = .I,
    team,
    scorer,
    recent_goals,
    recent_penalty_goals,
    total_intl_goals_in_file,
    expected_matches = round(expected_matches, 3),
    lambda_goals = round(lambda_goals, 3),
    golden_boot_prob = round(golden_boot_prob, 4),
    sim_mean_goals = round(sim_mean_goals, 3),
    prob_5plus_goals = round(prob_5plus_goals, 4),
    prob_6plus_goals = round(prob_6plus_goals, 4)
  )]
)

cat("\n==============================\n")
cat("DISTRIBUCIÓN DEL MÁXIMO DE GOLES\n")
cat("==============================\n")

print(max_goals_dist)

cat("\nRecomendación número de goles:", recommended_goals, "\n")

cat("\nArchivos guardados en:\n")
cat(out_dir, "\n")