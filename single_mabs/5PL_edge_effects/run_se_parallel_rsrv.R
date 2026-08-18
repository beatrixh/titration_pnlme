# Installed once here, sequentially, before any workers are spun up --
# install.packages() writes to a shared library path, so running it inside
# run_se_for_model() would have every worker try to install concurrently into
# the same location, risking a corrupted/partial install.
install.packages("/usr/local/Lixoft/MonolixSuite2024R1/connectors/lixoftConnectors.tar.gz",
                 repos = NULL, type="source", INSTALL_opts ="--no-multiarch")
install.packages("ps")

library(lixoftConnectors)
library(ps)
library(parallel)

models_dir <- "/home/bhaddock/repos/bnAb_pnlme/single_mabs/5PL_edge_effects/model_files"

# Fill in whichever already-fitted models you want SEs for.
model_names <- c('m182')

to_long_df <- function(x, value_col = "value") {
  if (is.data.frame(x)) return(x)
  df <- data.frame(parameter = names(x), value = as.numeric(x))
  names(df)[2] <- value_col
  df
}

run_se_for_model <- function(model_name, models_dir) {
  library(lixoftConnectors)
  initializeLixoftConnectors(software = "monolix", force = T,
                             path = "/usr/local/Lixoft/MonolixSuite2024R1/")
  library(ps)

  savedir <- file.path(models_dir, model_name)
  fitted_mlxtran <- file.path(savedir, paste0(model_name, "_fitted.mlxtran"))

  log_path <- file.path(models_dir, paste0(model_name, "_se_log.txt"))
  log_step <- function(step) {
    mem_mb <- round(as.numeric(ps::ps_memory_info(ps::ps_handle())["rss"]) / 1024^2, 1)
    cat(sprintf("[%s] %s :: %.1f MB\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), step, mem_mb),
        file = log_path, append = TRUE)
  }

  se_flag <- file.path(savedir, "_se_complete.flag")
  already_done <- file.exists(se_flag)

  if (!file.exists(fitted_mlxtran)) {
    message(sprintf("[%s] skipped: no fitted project at %s (model hasn't been fit yet)",
                     model_name, fitted_mlxtran))
    return(invisible(model_name))
  }

  if (already_done) {
    return(invisible(model_name))
  }

  start_time <- Sys.time()
  tryCatch({
    # Load the already-converged fit, not the original mlxtran -- this
    # skips straight to SE estimation without re-running SAEM. Reloading is
    # slower than computing SE in the same session right after a fresh fit
    # (~40% slower for the SE step itself, per the earlier timing test),
    # but still far cheaper than a full refit.
    loadProject(fitted_mlxtran)
    log_step(model_name)
    log_step("fitted project loaded")

    runStandardErrorEstimation()
    log_step("finished runStandardErrorEstimation")

    se <- getEstimatedStandardErrors()
    for (nm in names(se)) {
      write.csv(to_long_df(se[[nm]]), file.path(savedir, paste0("se_", nm, ".csv")), row.names = FALSE)
    }
    log_step("wrote se csvs")

    file.create(se_flag)
    elapsed <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 1)
    log_step(sprintf("COMPLETE (total runtime %.1f min)", elapsed))
  }, error = function(e) {
    elapsed <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 1)
    log_step(sprintf("FAILED after %.1f min: %s", elapsed, conditionMessage(e)))
    message(sprintf("[%s] failed: %s", model_name, conditionMessage(e)))
  })

  invisible(model_name)
}

# Skip anything that already has SEs computed.
already_done <- sapply(model_names, function(m) {
  file.exists(file.path(models_dir, m, "_se_complete.flag"))
})
model_names <- model_names[!already_done]

n_workers <- 1
cl <- makeCluster(n_workers)

results <- tryCatch(
  parLapplyLB(
    cl, model_names, run_se_for_model,
    models_dir = models_dir
  ),
  finally = stopCluster(cl)
)
