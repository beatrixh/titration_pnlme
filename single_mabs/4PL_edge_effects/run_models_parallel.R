library(lixoftConnectors)
library(ps)
library(parallel)
library(dplyr)

monolix_path <- "C:/Program Files/Lixoft/MonolixSuite2024R1"
# models_dir <- "C:/Users/bhaddock/repos/automated_run_models"
models_dir <- "C:/Users/bhaddock/repos/bnAb_pnlme/single_mabs/4PL_edge_effects/model_files"
model_files <- list.files(models_dir, pattern = "^4PL_edge_effects_m[0-9]+\\.mlxtran$")
model_names <- sub("^4PL_edge_effects_(m[0-9]+)\\.mlxtran$", "\\1", model_files)
model_names <- model_names[order(as.integer(sub("^m", "", model_names)))]

model_names <- c('m13','m5','m0')#,'m37','m1','m16','m50','m60')

run_one_model <- function(model_name, models_dir, monolix_path) {
  library(lixoftConnectors)
  library(dplyr)
  initializeLixoftConnectors(path = monolix_path)

  
  log_path <- file.path(models_dir, paste0(model_name, "_log.txt"))
  log_step <- function(step) {
    mem_mb <- round(as.numeric(ps::ps_memory_info(ps::ps_handle())["rss"]) / 1024^2, 1)
    cat(sprintf("[%s] %s :: %.1f MB\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), step, mem_mb),
        file = log_path, append = TRUE)
  }
  
  
  mlxtran_path <- file.path(models_dir, paste0("4PL_edge_effects_", model_name, ".mlxtran"))
  savedir <- file.path(models_dir, model_name)

  if (file.exists(mlxtran_path) && !dir.exists(savedir)) {
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
      
      pop <- getEstimatedPopulationParameters()
      ind <- getEstimatedIndividualParameters()
      loglik <- getEstimatedLogLikelihood()

      dir.create(savedir, recursive = TRUE)
      saveProject(file.path(savedir, paste0(model_name, "_fitted.mlxtran")))
      log_step("saved project")
      
      write.csv(pop, file.path(savedir, "pop.csv"), row.names = FALSE)
      for (nm in names(ind)) {
        write.csv(ind[[nm]], file.path(savedir, paste0("ind_", nm, ".csv")), row.names = FALSE)
      }
      write.csv(data.frame(as.list(unlist(loglik))), file.path(savedir, "loglik.csv"), row.names = FALSE)
      file.create(file.path(savedir, "_complete.flag"))
      log_step("COMPLETE")
    }, error = function(e) {
      log_step(sprintf("FAILED: %s", conditionMessage(e)))          # <- add
      message(sprintf("[%s] failed: %s", model_name, conditionMessage(e)))
    })
  }

  invisible(model_name)
}

n_workers <- 3
cl <- makeCluster(n_workers)

results <- tryCatch(
  parLapplyLB(
    cl, model_names, run_one_model,
    models_dir = models_dir, monolix_path = monolix_path
  ),
  finally = stopCluster(cl)
)
