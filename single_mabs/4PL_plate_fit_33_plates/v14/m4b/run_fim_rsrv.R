# Re-estimates the Fisher information matrix (standard errors) for an
# already-fitted model on rsrv, by linearization only (cheaper in memory and
# time than the stochastic approximation FIM).
#
# Monolix's FIM needs SAEM to have been run in the current session. The
# saved local results can't stand in for that (their .Internals/state.dat is
# empty, so loading them gives "SAEM must be launched before!"). Instead this
# warm-starts SAEM from the local estimates (<model_name>/populationParameters.txt)
# with a short run and no simulated annealing, so the estimates barely move,
# and then computes the FIM. Outputs go to a new folder so the local results
# are untouched; pop_estimate_drift.csv compares old vs new estimates.
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
out_name <- paste0(model_name, "_fim")

# Short warm-start SAEM from the converged estimates.
warm_exploratory_iterations <- 100
warm_smoothing_iterations <- 200

to_long_df <- function(x, value_col = "value") {
  if (is.data.frame(x)) return(x)
  df <- data.frame(parameter = names(x), value = as.numeric(x))
  names(df)[2] <- value_col
  df
}

mlxtran_path <- file.path(models_dir, paste0(model_name, ".mlxtran"))
local_pop_path <- file.path(models_dir, model_name, "populationParameters.txt")
savedir <- file.path(models_dir, out_name)
dir.create(savedir, showWarnings = FALSE)

log_path <- file.path(models_dir, paste0(out_name, "_log.txt"))
log_step <- function(step) {
  mem_mb <- round(as.numeric(ps::ps_memory_info(ps::ps_handle())["rss"]) / 1024^2, 1)
  cat(sprintf("[%s] %s :: %.1f MB\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), step, mem_mb),
      file = log_path, append = TRUE)
}

write_se <- function(suffix) {
  se <- getEstimatedStandardErrors()
  if (is.null(se)) stop(sprintf("no standard errors returned (%s)", suffix))
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

  # Start from the local estimates. Fixed parameters stay fixed; everything
  # else keeps its estimation method.
  local_pop <- read.csv(local_pop_path, stringsAsFactors = FALSE)
  info <- getPopulationParameterInformation()
  missing <- setdiff(info$name, local_pop$parameter)
  if (length(missing) > 0) {
    stop(sprintf("no local estimate for: %s", paste(missing, collapse = ", ")))
  }
  info$initialValue <- local_pop$value[match(info$name, local_pop$parameter)]
  setPopulationParameterInformation(info)
  log_step("initial values set to local estimates")

  setProjectSettings(directory = savedir)
  setPopulationParameterEstimationSettings(
    nbexploratoryiterations = warm_exploratory_iterations,
    nbsmoothingiterations = warm_smoothing_iterations,
    exploratoryautostop = FALSE,
    smoothingautostop = FALSE,
    simulatedannealing = FALSE
  )
  saveProject(file.path(models_dir, paste0(out_name, ".mlxtran")))

  log_step("starting warm-start runPopulationParameterEstimation")
  runPopulationParameterEstimation()
  log_step("finished warm-start runPopulationParameterEstimation")

  new_pop <- getEstimatedPopulationParameters()
  drift <- data.frame(parameter = names(new_pop),
                      local = local_pop$value[match(names(new_pop), local_pop$parameter)],
                      warm_start = as.numeric(new_pop))
  drift$abs_diff <- abs(drift$warm_start - drift$local)
  write.csv(drift, file.path(savedir, "pop_estimate_drift.csv"), row.names = FALSE)
  log_step(sprintf("max |estimate drift| = %.4g", max(drift$abs_diff, na.rm = TRUE)))

  # Linearization FIM is evaluated at the conditional modes (EBEs).
  log_step("starting runConditionalModeEstimation")
  runConditionalModeEstimation()
  log_step("finished runConditionalModeEstimation")

  log_step("starting runStandardErrorEstimation (lin)")
  runStandardErrorEstimation(linearization = TRUE)
  log_step("finished runStandardErrorEstimation (lin)")
  write_se("lin")

  saveProject(file.path(models_dir, paste0(out_name, ".mlxtran")))
  log_step("saved project")

  file.create(file.path(savedir, "_fim_complete.flag"))
  elapsed <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 1)
  log_step(sprintf("COMPLETE (total runtime %.1f min)", elapsed))
}, error = function(e) {
  elapsed <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 1)
  log_step(sprintf("FAILED after %.1f min: %s", elapsed, conditionMessage(e)))
  message(sprintf("[%s] failed: %s", model_name, conditionMessage(e)))
})
