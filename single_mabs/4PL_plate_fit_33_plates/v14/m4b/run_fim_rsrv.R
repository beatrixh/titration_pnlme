# Re-estimates the Fisher information matrix (standard errors) for an
# already-fitted model on rsrv, without re-running SAEM.
#
# Loading the original .mlxtran also reloads the previous results from its
# exportpath folder (<model_name>/.Internals), so the population estimates
# from the local run are reused. Copy that whole results folder to rsrv
# alongside the .mlxtran.
#
# Launch from anywhere (paths are absolute):
#   nohup Rscript run_fim_rsrv.R > run_fim_rsrv.Rout 2>&1 &

# Installed once here, sequentially -- see run_models_parallel_rsrv.R.
install.packages("/usr/local/Lixoft/MonolixSuite2024R1/connectors/lixoftConnectors.tar.gz",
                 repos = NULL, type="source", INSTALL_opts ="--no-multiarch")

library(lixoftConnectors)
library(ps)
initializeLixoftConnectors(software = "monolix", force = T,
                           path = "/usr/local/Lixoft/MonolixSuite2024R1/")

models_dir <- "/home/bhaddock/repos/titration_pnlme/single_mabs/4PL_plate_fit_33_plates/v14/m4b/model_files"
model_name <- "m3_larger_logit_alpha_rerun"

# Stochastic approximation FIM settings. The local run used maxiterations = 5000.
fim_miniterations <- 100
fim_maxiterations <- 10000
# Also compute the linearization FIM (fast; useful cross-check of the SA one).
also_linearization <- TRUE

to_long_df <- function(x, value_col = "value") {
  if (is.data.frame(x)) return(x)
  df <- data.frame(parameter = names(x), value = as.numeric(x))
  names(df)[2] <- value_col
  df
}

mlxtran_path <- file.path(models_dir, paste0(model_name, ".mlxtran"))
savedir <- file.path(models_dir, model_name)

log_path <- file.path(models_dir, paste0(model_name, "_fim_log.txt"))
log_step <- function(step) {
  mem_mb <- round(as.numeric(ps::ps_memory_info(ps::ps_handle())["rss"]) / 1024^2, 1)
  cat(sprintf("[%s] %s :: %.1f MB\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), step, mem_mb),
      file = log_path, append = TRUE)
}

write_se <- function(suffix) {
  se <- getEstimatedStandardErrors()
  for (nm in names(se)) {
    write.csv(to_long_df(se[[nm]]),
              file.path(savedir, paste0("se_", nm, "_", suffix, ".csv")), row.names = FALSE)
  }
}

start_time <- Sys.time()
tryCatch({
  loadProject(mlxtran_path)
  log_step(model_name)
  log_step("project loaded")

  if (is.null(getEstimatedPopulationParameters())) {
    stop("no population estimates loaded -- was the results folder (incl. .Internals) copied over?")
  }

  # Keep the previous (local) SA FIM output rather than overwriting it.
  fim_dir <- file.path(savedir, "FisherInformation")
  if (dir.exists(fim_dir)) {
    backup <- file.path(savedir, paste0("FisherInformation_local_", format(Sys.time(), "%Y%m%d_%H%M%S")))
    file.rename(fim_dir, backup)
    log_step(sprintf("backed up previous FIM to %s", basename(backup)))
  }

  setStandardErrorEstimationSettings(miniterations = fim_miniterations,
                                     maxiterations = fim_maxiterations)

  log_step("starting runStandardErrorEstimation (SA)")
  runStandardErrorEstimation(linearization = FALSE)
  log_step("finished runStandardErrorEstimation (SA)")
  write_se("sa")

  if (also_linearization) {
    log_step("starting runStandardErrorEstimation (lin)")
    runStandardErrorEstimation(linearization = TRUE)
    log_step("finished runStandardErrorEstimation (lin)")
    write_se("lin")
  }

  saveProject(file.path(savedir, paste0(model_name, "_fim.mlxtran")))
  log_step("saved project")

  file.create(file.path(savedir, "_fim_complete.flag"))
  elapsed <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 1)
  log_step(sprintf("COMPLETE (total runtime %.1f min)", elapsed))
}, error = function(e) {
  elapsed <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 1)
  log_step(sprintf("FAILED after %.1f min: %s", elapsed, conditionMessage(e)))
  message(sprintf("[%s] failed: %s", model_name, conditionMessage(e)))
})
