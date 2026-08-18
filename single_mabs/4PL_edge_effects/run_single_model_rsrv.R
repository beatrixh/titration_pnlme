# server
install.packages("/usr/local/Lixoft/MonolixSuite2024R1/connectors/lixoftConnectors.tar.gz",
                 repos = NULL, type="source", INSTALL_opts ="--no-multiarch")
install.packages("ps")
library(lixoftConnectors)
initializeLixoftConnectors(software = "monolix", force = T,
                           path = "/usr/local/Lixoft/MonolixSuite2024R1/")

library(dplyr)
library(ps)


model_name <- "m0"
models_dir <- "model_files/"

run_model_singly <- function(model_name, models_dir) {

  log_path <- file.path(models_dir, paste0(model_name, "_log.txt"))
  log_step <- function(step) {
    mem_mb <- round(as.numeric(ps::ps_memory_info(ps::ps_handle())["rss"]) / 1024^2, 1)
    cat(sprintf("[%s] %s :: %.1f MB\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), step, mem_mb),
        file = log_path, append = TRUE)
  }
  
  
  mlxtran_path <- file.path(models_dir, paste0("4PL_edge_effects_", model_name, ".mlxtran"))
  savedir <- file.path(models_dir, model_name)

  already_done <- file.exists(file.path(savedir, "_complete.flag"))

  if (file.exists(mlxtran_path) && !already_done) {
    tryCatch({
      loadProject(mlxtran_path)
      log_step(model_name)
      log_step("project loaded")

      autoInitValues <- getFixedEffectsByAutoInit()
      setPopulationParameterInformation(autoInitValues)
      log_step("initial values configured")

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
      # runStandardErrorEstimation()
      # log_step("finished runStandardErrorEstimation")
      
      pop <- getEstimatedPopulationParameters()
      ind <- getEstimatedIndividualParameters()
      loglik <- getEstimatedLogLikelihood()
      # se <- runStandardErrorEstimation()


      dir.create(savedir, recursive = TRUE)
      saveProject(file.path(savedir, paste0(model_name, "_fitted.mlxtran")))
      log_step("saved project")
      
      file.create(file.path(savedir, "_complete.flag"))
      log_step("COMPLETE")
    }, error = function(e) {
      log_step(sprintf("FAILED: %s", conditionMessage(e)))
      message(sprintf("[%s] failed: %s", model_name, conditionMessage(e)))
    })
  }

  invisible(model_name)
}

# Run and time the estimation scenario
elapsed <- system.time({
  run_model_singly(model_name, models_dir)
})

cat("\n--- Timing ---\n")
print(elapsed)
cat(sprintf("\nElapsed (wall clock): %.2f seconds\n", elapsed["elapsed"]))