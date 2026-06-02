suppressPackageStartupMessages({
  library(data.table)
})

base_dir <- "resultados_server/v6_multiseed_consolidated"

tags <- paste0("MONTE_CARLO_SENSITIVITY_SEED_", 1001:1010)

out_dir <- file.path(base_dir, "MONTE_CARLO_SENSITIVITY_AVERAGED_10SEEDS")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

read_seed_file <- function(tag, fname) {
  path <- file.path(base_dir, tag, fname)
  if (!file.exists(path)) {
    stop("No existe: ", path)
  }
  x <- fread(path)
  x[, mc_tag := tag]
  x
}

# ------------------------------------------------------------
# 1. Promedio probabilidades por etapa
# ------------------------------------------------------------

stage_all <- rbindlist(lapply(tags, read_seed_file,
                              fname = "mc_team_stage_probabilities_v6MS_official_annexC.csv"
), fill = TRUE)

stage_avg <- stage_all[
  ,
  .(
    qualified_R32_mean = mean(qualified_R32),
    qualified_R32_sd   = sd(qualified_R32),
    
    reached_R16_mean = mean(reached_R16),
    reached_R16_sd   = sd(reached_R16),
    
    reached_QF_mean = mean(reached_QF),
    reached_QF_sd   = sd(reached_QF),
    
    reached_SF_mean = mean(reached_SF),
    reached_SF_sd   = sd(reached_SF),
    
    reached_Final_mean = mean(reached_Final),
    reached_Final_sd   = sd(reached_Final),
    
    champion_mean = mean(champion),
    champion_sd   = sd(champion)
  ),
  by = team
][order(-champion_mean, -reached_Final_mean, -reached_SF_mean)]

fwrite(stage_avg, file.path(out_dir, "mc_team_stage_probabilities_AVERAGED_10SEEDS.csv"))

# ------------------------------------------------------------
# 2. Promedio posiciones de grupo
# ------------------------------------------------------------

pos_all <- rbindlist(lapply(tags, read_seed_file,
                            fname = "mc_group_position_probabilities_v6MS_official_annexC.csv"
), fill = TRUE)

pos_avg <- pos_all[
  ,
  .(
    group = first(group),
    
    p_pos1_mean = mean(p_pos1),
    p_pos1_sd   = sd(p_pos1),
    
    p_pos2_mean = mean(p_pos2),
    p_pos2_sd   = sd(p_pos2),
    
    p_pos3_mean = mean(p_pos3),
    p_pos3_sd   = sd(p_pos3),
    
    p_pos4_mean = mean(p_pos4),
    p_pos4_sd   = sd(p_pos4),
    
    p_qualified_R32_mean = mean(p_qualified_R32),
    p_qualified_R32_sd   = sd(p_qualified_R32),
    
    avg_points_mean = mean(avg_points),
    avg_gd_mean     = mean(avg_gd),
    avg_gf_mean     = mean(avg_gf)
  ),
  by = team
][order(group, -p_pos1_mean, -p_pos2_mean, -p_qualified_R32_mean)]

fwrite(pos_avg, file.path(out_dir, "mc_group_position_probabilities_AVERAGED_10SEEDS.csv"))

# ------------------------------------------------------------
# 3. Campeón promedio
# ------------------------------------------------------------

champ_all <- rbindlist(lapply(tags, read_seed_file,
                              fname = "mc_simulation_champions_v6MS_official_annexC.csv"
), fill = TRUE)

champ_avg <- champ_all[
  ,
  .(
    p_champion_mean = mean(p_champion),
    p_champion_sd   = sd(p_champion),
    min_p_champion  = min(p_champion),
    max_p_champion  = max(p_champion)
  ),
  by = champion
][order(-p_champion_mean)]

fwrite(champ_avg, file.path(out_dir, "mc_simulation_champions_AVERAGED_10SEEDS.csv"))

# ------------------------------------------------------------
# 4. Resumen consola
# ------------------------------------------------------------

cat("\n====================================\n")
cat("PROMEDIO SENSIBILIDAD 10 MC_SEEDS\n")
cat("====================================\n")

cat("\nTop campeón promedio:\n")
print(champ_avg[1:15])

cat("\nTop llegar a final promedio:\n")
print(stage_avg[order(-reached_Final_mean)][1:15, .(
  team,
  reached_Final_mean,
  reached_Final_sd,
  champion_mean,
  champion_sd
)])

cat("\nArchivos generados en:\n")
cat(out_dir, "\n")