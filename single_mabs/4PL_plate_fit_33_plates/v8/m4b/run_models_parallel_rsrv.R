# Installed once here, sequentially, before any workers are spun up --
# install.packages() writes to a shared library path, so running it inside
# run_one_model() would have every worker try to install concurrently into
# the same location, risking a corrupted/partial install.
install.packages("/usr/local/Lixoft/MonolixSuite2024R1/connectors/lixoftConnectors.tar.gz",
                 repos = NULL, type="source", INSTALL_opts ="--no-multiarch")

library(lixoftConnectors)
library(ps)
library(parallel)
library(dplyr)

models_dir <- "/home/bhaddock/repos/titration_pnlme/single_mabs/4PL_plate_fit_33_plates/v8/m4b/model_files"
model_files <- list.files(models_dir, pattern = "^m[0-9]+\\.mlxtran$")
model_names <- sub("^(m[0-9]+)\\.mlxtran$", "\\1", model_files)
model_names <- model_names[order(as.integer(sub("^m", "", model_names)))]

model_names <- paste0("m", 1:512)
# Top 100 4PL_plate_fit_small_data v8 models by BICc (see combined_likelihood_report_4PL_5PL.csv)
model_names <- c("m414", "m407", "m472", "m189", "m424", "m471", "m480", "m165", "m406", "m495",
                  "m429", "m493", "m479", "m470", "m408", "m415", "m432", "m486", "m245", "m342",
                  "m421", "m422", "m423", "m487", "m416", "m101", "m485", "m503", "m114", "m502",
                  "m440", "m177", "m413", "m496", "m510", "m181", "m237", "m229", "m50", "m417",
                  "m478", "m113", "m473", "m447", "m185", "m241", "m494", "m178", "m501", "m337",
                  "m143", "m104", "m511", "m166", "m410", "m249", "m124", "m186", "m509", "m253",
                  "m431", "m122", "m242", "m121", "m230", "m57", "m175", "m430", "m250", "m150",
                  "m173", "m490", "m338", "m231", "m168", "m439", "m109", "m238", "m360", "m375",
                  "m489", "m469", "m169", "m481", "m366", "m418", "m95", "m383", "m179", "m37",
                  "m343", "m123", "m49", "m438", "m504", "m170", "m233", "m426", "m22", "m477")

run_one_model <- function(model_name, models_dir) {
  library(lixoftConnectors)
  initializeLixoftConnectors(software = "monolix", force = T,
                             path = "/usr/local/Lixoft/MonolixSuite2024R1/")

  library(dplyr)
  library(ps)


  log_path <- file.path(models_dir, paste0(model_name, "_log.txt"))
  log_step <- function(step) {
    mem_mb <- round(as.numeric(ps::ps_memory_info(ps::ps_handle())["rss"]) / 1024^2, 1)
    cat(sprintf("[%s] %s :: %.1f MB\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), step, mem_mb),
        file = log_path, append = TRUE)
  }


  mlxtran_path <- file.path(models_dir, paste0(model_name, ".mlxtran"))
  savedir <- file.path(models_dir, model_name)

  already_done <- file.exists(file.path(savedir, "_complete.flag"))

  if (file.exists(mlxtran_path) && !already_done) {
    start_time <- Sys.time()
    tryCatch({
      loadProject(mlxtran_path)
      log_step(model_name)
      log_step("project loaded")

      # autoInitValues <- getFixedEffectsByAutoInit()
      # setPopulationParameterInformation(autoInitValues)
      # log_step("initial values configured")

      # popParams <- getPopulationParameterInformation()

      # betaRows <- grepl("^beta_", popParams$name)
      # popParams$initialValue[betaRows] <- 0

      # omegaRows <- grepl("^omega_", popParams$name)
      # popParams$initialValue[omegaRows] <- 1

      # popParams <- popParams %>%
      #   rows_update(autoInitValues, by = "name")
      # setPopulationParameterInformation(popParams)

      # defaults <- c(a = 1, b = 0.3, c = 1)
      # for (nm in names(defaults)) {
      #   if (nm %in% popParams$name) {
      #     popParams$initialValue[popParams$name == nm] <- defaults[nm]
      #   }
      # }
      # setPopulationParameterInformation(popParams)

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
      elapsed <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 1)
      log_step(sprintf("COMPLETE (total runtime %.1f min)", elapsed))
    }, error = function(e) {
      elapsed <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 1)
      log_step(sprintf("FAILED after %.1f min: %s", elapsed, conditionMessage(e)))
      message(sprintf("[%s] failed: %s", model_name, conditionMessage(e)))
    })
  }

  invisible(model_name)
}

n_workers <- 1 
cl <- makeCluster(n_workers)

results <- tryCatch(
  parLapplyLB(
    cl, model_names, run_one_model,
    models_dir = models_dir
  ),
  finally = stopCluster(cl)
)
