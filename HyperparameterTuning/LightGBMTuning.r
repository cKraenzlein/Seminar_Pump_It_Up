# Load required packages
library(mlr3)
library(mlr3viz)
library(mlr3tuning)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3pipelines)

# Set the working directory
here::here()

# Load the training data
source("./Data/CreateTrainingData.R")
Data <- Data_Training_Final

# Set seed for reproducibility
set.seed(24)

# Pipeline for imputing missing values (excluding population, gps_height, construction_year, amount_tsh)
RF_imputation_regr = lrn("regr.ranger",num.threads = 8, num.trees = 250,min.node.size = 5,mtry = 6,max.depth = 10)
RF_imputation_classif = lrn("classif.ranger",num.threads = 8, num.trees = 250,min.node.size = 5,mtry = 6,max.depth = 10)

flag_missing = po("missind", affect_columns = selector_type(c("numeric", "integer", "logical", "factor")))
imp_num = po("imputelearner", learner = RF_imputation_regr, affect_columns = selector_name(c("longitude", "latitude")), id = "impute_num")
imp_factor = po("imputelearner", learner = RF_imputation_classif, affect_columns = selector_type("factor"), id = "impute_factor")
imp_bin = po("imputelearner", learner = RF_imputation_classif, affect_columns = selector_type("logical"), id = "impute_bin")
po_select = po("select", selector = selector_invert(selector_name(c("missing_population_log", "missing_gps_height", "missing_longitude", "missing_construction_year", "missing_amount_tsh_log"))))

imp_sel <- gunion(list(flag_missing, imp_num, imp_factor, imp_bin)) %>>% po("featureunion") %>>% po_select

logical_to_int = po("colapply", id = "encode_logical", applicator = as.numeric, affect_columns = selector_type("logical"))

graph = imp_sel %>>% logical_to_int

graph$plot()

# Define task
task = as_task_classif(Data, id = "PumpItUp", target = "status_group")

# For stratified sampling
task$set_col_roles("status_group", c("target", "stratum"))

# Define Learner
learner = lrn("classif.lightgbm", id = "lightGBM", num_threads = 8)
# Combine graph and learner
graph_learner = as_learner(graph %>>% learner)

# Define the hyperparameter space
hyper_param_space <- ps(
    lightGBM.learning_rate =               p_dbl(exp(-3), 1, logscale = TRUE),
    lightGBM.num_leaves =                  p_int(2, 256, logscale = TRUE),
    lightGBM.feature_fraction =            p_dbl(0.5, 1),
    lightGBM.bagging_fraction =            p_dbl(0.5, 1),
    lightGBM.min_sum_hessian_in_leaf =     p_dbl(exp(-7), exp(3), logscale = TRUE),
    lightGBM.min_data_in_leaf =            p_int(1, round(exp(6)), logscale = TRUE)
)
# Resampling and Measure Strategy
resampling <- rsmp("cv", folds = 3)
measure    <- msr("classif.acc")

# Define Tuner and Termination
tuner_rs <- tnr("random_search")
term_rs  <- trm("combo", list(trm("clock_time", stop_time = Sys.time() + 4 * 3600),
                              trm("evals", n_evals = 100)), any = TRUE)

# Tuning instance
tuning_instance <- ti(
  task       = task,
  learner    = graph_learner,
  resampling = resampling,
  measures   = measure,
  terminator = term_rs,
  search_space  = hyper_param_space
)

# Start tuning
tuner_rs$optimize(tuning_instance)

graph_learner$param_set$data
# Save visualization of the tuning process
source("./HyperparameterTuning/VizHyper.r")
Plot_Tuning_LightGBM(tuning_instance)

# Save tuning results
tuning_instance$resampling <- NULL
tuning_instance$history <- NULL

future::plan("multisession", workers = 8)
binary_tuning_results <- serialize(tuning_instance, NULL)
# Define the chunk size (90 MB = 90 * 1024 * 1024 bytes)
chunk_size_bytes <- 90 * 1024^2
# Get the total size of the binary object
total_size <- length(binary_tuning_results)
# Calculate the number of chunks
num_chunks <- ceiling(total_size / chunk_size_bytes)

chunks <- split(binary_tuning_results, ceiling(seq_along(binary_tuning_results) / (total_size / num_chunks)))

for(i in seq_along(chunks)) {
  saveRDS(chunks[[i]], file = paste0("./HyperparameterTuning/Results/LightGBM_tuning_data_part", i, ".rds"))
}

saveRDS(tuning_instance, "./HyperparameterTuning/Results/LightGBM_tuning_data.rds", compress = "xz")

# Display results
best_rs <- tuning_instance$result_learner_param_vals
cat(sprintf("Stage 1 (Random): acc=%.4f | %s\n",
            tuning_instance$result_y,
            paste(sprintf("%s=%s", names(best_rs), unlist(best_rs)), collapse = ", ")))

# Fit and Save Final Model
final_params <- tuning_instance$result_learner_param_vals

graph_learner$param_set$values <- final_params

graph_learner$train(task)
# Get the in Training Performance
performance_train = graph_learner$predict(task)$score(msr("classif.acc"))
print(performance_train)

# Plot the feature importance
plot_importance(graph_learner$importance(), "LightGBM")

# Reduce size of the stored model
graph_learner$task <-         NULL  # Remove training data
graph_learner$resampling <-   NULL  # Remove resampling results
graph_learner$results <-      NULL  # Remove evaluation results
graph_learner$history <-      NULL  # Remove tuning history
graph_learner$archive <-      NULL  # Remove archived models from the tuning process
graph_learner$metadata <-     NULL  # Remove metadata

saveRDS(graph_learner, "./FinalModels/lightGBM_final_model.rds", compress = "gzip")
cat("lightGBM_final_model.rds gespeichert.")
