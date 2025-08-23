# Load required packages
library(mlr3)
library(mlr3viz)
library(mlr3tuning) 
library(mlr3extralearners)
library(mlr3learners)
library(mlr3pipelines)

# Set the working directory
here::here()

# Load the training data
source("./Data/CreateTrainingData.R")
Data <- Data_Training_Final

# Set seed for reproducibility
set.seed(24)

# Pipeline
RF_imputation_regr = lrn("regr.ranger",num.threads = 8, num.trees = 250,min.node.size = 5,mtry = 6,max.depth=10)
RF_imputation_classif = lrn("classif.ranger",num.threads = 8, num.trees = 250,min.node.size = 5,mtry = 6,max.depth=10)

flag_missing = po("missind")
imp_num = po("imputelearner", learner = RF_imputation_regr, affect_columns = selector_name(c("longitude", "latitude")), id = "impute_num")
imp_factor = po("imputelearner", learner = RF_imputation_classif, affect_columns = selector_type("factor"), id = "impute_factor")
imp_bin = po("imputelearner", learner = RF_imputation_classif, affect_columns = selector_type("logical"), id = "impute_bin")
po_select = po("select", selector = selector_invert(selector_name(c("missing_population_log", "missing_gps_height", "missing_longitude", "missing_construction_year", "missing_amount_tsh_log"))))

imp_sel <- flag_missing %>>% list(imp_num, imp_factor, imp_bin) %>>% po("featureunion") %>>% po_select

encode_factors = po("encode", id = "encode_factors", method = "one-hot", affect_columns = selector_type("factor"))
logical_to_int = po("colapply", id = "encode_logical", applicator = as.integer, affect_columns = selector_type("logical"))

graph = imp_sel %>>% encode_factors %>>% logical_to_int

graph$plot()

# Define task
task = as_task_classif(Data, id = "PumpItUp", target = "status_group")

# For stratified sampling
task$set_col_roles("status_group", c("target", "stratum"))

# Define Learner
learner = lrn("classif.catboost", id = "catboost", thread_count = 8)

# Combine graph and learner
graph_learner = as_learner(graph %>>% learner)
View(graph_learner$param_set$data)
# Hyperparameter space
hyper_param_space <- ps(
  catboost.iterations    = p_int(100, 1500),
  ccatboost.learning_rate = p_dbl(0.01, 0.3),
  catboost.depth         = p_int(4, 10),
  catboost.l2_leaf_reg   = p_dbl(1, 10)
)

# Resampling and Measure Strategy
resampling <- rsmp("cv", folds = 5)
measure    <- msr("classif.acc")

# Define Tuner and Termination
tuner_rs <- tnr("random_search")
term_rs  <- trm("combo", list(trm("clock_time", stop_time = Sys.time() + 1 * 3600),
                              trm("evals", n_evals = 5)), any = TRUE)

# Tuning instance
tuning_instance <- ti(
  task       = task,
  learner    = graph_learner,
  resampling = resampling,
  measures   = measure,
  terminator = term_rs
)

# Start tuning
tuner_rs$optimize(tuning_instance)

# Save tuning results
saveRDS(tuning_instance, "catboost_tuning_data.rds")

# Display results
best_rs <- tuning_instance$result_learner_param_vals
cat(sprintf("Stage 1 (Random): acc=%.4f | %s\n",
            tuning_instance$result_y,
            paste(sprintf("%s=%s", names(best_rs), unlist(best_rs)), collapse = ", ")))

# Fit and Save Final Model
final_params <- tuning_instance$result_learner_param_vals

learner$param_set$values <- final_params
learner$train(task)

saveRDS(learner, "catboost_final_model.rds")
cat("catboost_final_model.rds gespeichert\n")