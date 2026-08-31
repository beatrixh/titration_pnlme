# Usage: Rscript run_single_model_rsrv.R /path/to/some_model.mlxtran

install.packages("/usr/local/Lixoft/MonolixSuite2024R1/connectors/lixoftConnectors.tar.gz",
                 repos = NULL, type="source", INSTALL_opts ="--no-multiarch")
install.packages("ps")

library(lixoftConnectors)
library(ps)
library(dplyr)

mlxtran_path <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(mlxtran_path) || mlxtran_path == "") {
  stop("usage: Rscript run_single_model_rsrv.R /path/to/some_model.mlxtran")
}
if (!file.exists(mlxtran_path)) {
  stop(sprintf("no such file: %s", mlxtran_path))
}

models_dir <- dirname(mlxtran_path)

# Derives the model name from the file's own basename, stripping a trailing
# "..._m<N>" pattern if present (e.g. "5PL_edge_effects_m0.mlxtran" -> "m0"),
# so output lands in the same model_files/m<N>/ layout the rest of the
# pipeline (SE script, likelihood report) expects. Falls back to the full
# basename for files that don't follow that naming convention.
base_name <- tools::file_path_sans_ext(basename(mlxtran_path))
trailing_m <- regmatches(base_name, regexpr("m[0-9]+$", base_name))
model_name <- if (length(trailing_m) > 0 && nzchar(trailing_m)) trailing_m else base_name

initializeLixoftConnectors(software = "monolix", force = T,
                           path = "/usr/local/Lixoft/MonolixSuite2024R1/")

savedir <- file.path(models_dir, model_name)
log_path <- file.path(models_dir, paste0(model_name, "_log.txt"))
log_step <- function(step) {
  mem_mb <- round(as.numeric(ps::ps_memory_info(ps::ps_handle())["rss"]) / 1024^2, 1)
  cat(sprintf("[%s] %s :: %.1f MB\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), step, mem_mb),
      file = log_path, append = TRUE)
}

already_done <- file.exists(file.path(savedir, "_complete.flag"))
if (already_done) {
  message(sprintf("[%s] already complete (found %s/_complete.flag) -- nothing to do",
                   model_name, savedir))
  quit(status = 0)
}

start_time <- Sys.time()
tryCatch({
  loadProject(mlxtran_path)
  log_step(model_name)
  log_step("project loaded")

  autoInitValues <- getFixedEffectsByAutoInit()
  setPopulationParameterInformation(autoInitValues)

  popParams <- getPopulationParameterInformation()

  betaRows <- grepl("^beta_", popParams$name)
  popParams$initialValue[betaRows] <- 0

  omegaRows <- grepl("^omega_", popParams$name)
  popParams$initialValue[omegaRows] <- 1

  popParams <- popParams %>%
    rows_update(autoInitValues, by = "name")
  setPopulationParameterInformation(popParams)

  defaults <- c(a = 1, b = 0.3, c = 1)
  for (nm in names(defaults)) {
    if (nm %in% popParams$name) {
      popParams$initialValue[popParams$name == nm] <- defaults[nm]
    }
  }
  setPopulationParameterInformation(popParams)

  setConditionalModeEstimationSettings(
    nboptimizationiterationsmode = 2000
  )

  log_step("starting runPopulationParameterEstimation")
  runPopulationParameterEstimation()
  log_step("finished runPopulationParameterEstimation")
  runConditionalModeEstimation()
  log_step("finished runConditionalModeEstimation")
  runLogLikelihoodEstimation()
  log_step("finished runLogLikelihoodEstimation")

  pop <- getEstimatedPopulationParameters()
  ind <- getEstimatedIndividualParameters()
  loglik <- getEstimatedLogLikelihood()

  dir.create(savedir, recursive = TRUE, showWarnings = FALSE)
  saveProject(file.path(savedir, paste0(model_name, "_fitted.mlxtran")))
  log_step("saved project")

  to_long_df <- function(x, value_col = "value") {
    if (is.data.frame(x)) return(x)
    df <- data.frame(parameter = names(x), value = as.numeric(x))
    names(df)[2] <- value_col
    df
  }

  write.csv(to_long_df(pop, "value"), file.path(savedir, "pop.csv"), row.names = FALSE)
  for (nm in names(ind)) {
    write.csv(ind[[nm]], file.path(savedir, paste0("ind_", nm, ".csv")), row.names = FALSE)
  }
  write.csv(data.frame(as.list(unlist(loglik))), file.path(savedir, "loglik.csv"), row.names = FALSE)
  file.create(file.path(savedir, "_complete.flag"))

  elapsed <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 1)
  log_step(sprintf("COMPLETE (total runtime %.1f min)", elapsed))
  message(sprintf("[%s] COMPLETE (%.1f min)", model_name, elapsed))
}, error = function(e) {
  elapsed <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 1)
  log_step(sprintf("FAILED after %.1f min: %s", elapsed, conditionMessage(e)))
  message(sprintf("[%s] failed after %.1f min: %s", model_name, elapsed, conditionMessage(e)))
})
