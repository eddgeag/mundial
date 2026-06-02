suppressPackageStartupMessages({
  library(xgboost)
})

cat("xgboost servidor:", as.character(packageVersion("xgboost")), "\n")

base_multiseed_dir <- "resultados_server/v6_multiseed"

seed_dirs <- list.dirs(base_multiseed_dir, recursive = FALSE, full.names = TRUE)
seed_dirs <- seed_dirs[grepl("seed_[0-9]+$", seed_dirs)]

for (sd in seed_dirs) {
  old_file <- file.path(sd, "model_xgboost_v6_tuned.xgb")
  new_file <- file.path(sd, "model_xgboost_v6_tuned.ubj")

  cat("\nConvirtiendo:\n", old_file, "\n")

  if (!file.exists(old_file)) {
    warning("No existe: ", old_file)
    next
  }

  model <- xgb.load(old_file)
  xgb.save(model, new_file)

  cat("Guardado:\n", new_file, "\n")
}
