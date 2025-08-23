# Load required packages
library(mlr3)
library(mlr3learners)   # Provides learners like 'classif.ranger'
library(mlr3filters)    # For feature selection
library(mlr3viz)        # For plotting
library(mlr3tuning)     # For hyperparameter tuning
library(ranger)         # The underlying Random Forest implementation
library(mlr3pipelines)  # For pipelines
library(ggplot2)        # For plotting
library(dplyr)          # For data manipulation
library(patchwork)      # Combining Plots
# Source R file for training data
here::here()
# --------------------------------------------------------------------------------------------------------------------------
# Load the training data
source("CreateTrainingData.R")
Data <- Data_Training_Final
# --------------------------------------------------------------------------------------------------------------------------
# Load the test data
source("CreateTestDataSet.R")
ID <- Id
Test_Data <- Data_Test_Final
#---------------------------------------------------------------------------------------------------------------------------
# Set seed for reproducibility
set.seed(24)
# --------------------------------------------------------------------------------------------------------------------------
# Imputation graph
imp_num = list(po("missind"), 
               po("imputelearner", learner = lrn("regr.ranger",num.threads = 8, num.trees = 250,min.node.size = 5,mtry = 6), affect_columns = selector_name(c("longitude", "latitude")), id = "impute_num"))
imp_factor <- po("imputelearner", learner = lrn("classif.ranger", num.threads = 8, num.trees = 250,min.node.size = 5,mtry = 6), affect_columns = selector_type("factor"), id = "impute_factor")
imp_bin <- po("imputelearner", learner = lrn("classif.ranger", num.threads = 8, num.trees = 250,min.node.size = 5,mtry = 6), affect_columns = selector_type("logical"), id = "impute_bin")
po_select = po("select", selector = selector_invert(selector_name(c("missing_population_log", "missing_gps_height", "missing_longitude", "missing_construction_year", "missing_amount_tsh_log"))))

imp_sel <- imp_num %>>% po("featureunion") %>>% imp_factor %>>% imp_bin %>>% po_select
imp_sel$plot()
# --------------------------------------------------------------------------------------------------------------------------
# Random Forest with mlr3, Runtime approximately 2h 30min
future::plan("multisession", workers = 3)
# Define the task
task_RF_Hyper <- as_task_classif(Data, target = "status_group")
task_RF_Hyper$set_col_roles("status_group", c("target", "stratum"))
# Define the learner and the tuning spaces
#learner_RF = as_learner(imp_sel %>>% po(
learner_RF = lrn("classif.ranger",
  num.trees  = to_tune(p_int(500,2000)),
  mtry = to_tune(p_int(3,16)),
  max.depth = to_tune(p_int(15, 50)),
  min.node.size = to_tune(p_int(1,11)),
  splitrule = "gini",
<<<<<<< HEAD
  num.threads = 8
)
#))
=======
  num.threads = 5
)))
>>>>>>> c68eeb6a1e9656c00a9498b104f0e1f396e8d52d
# Define the tuning instance
instance_RF = ti(
  task = task_RF_Hyper,
  learner = learner_RF,
  resampling = rsmp("cv", folds = 3),
  measures = msr("classif.acc"),
  terminator = trm("combo", list(trm("clock_time", stop_time = Sys.time() + 1 * 3600),
                                 trm("evals", n_evals = 4)), any = TRUE)
)
tuner_RF = tnr("random_search")
# Tune the hyperparameters
tuner_RF$optimize(instance_RF)
# Result:
instance_RF$result$learner_param_vals
# With NAs
# max.depth = 25
# min.node.size = 9
# mtry = 8
# num.trees = 636
# splitrule = "gini"
# -----------------------
# Imputation excluding(construction_year, gps_height, amount_tsh_log, popultion_log)
# max.depth = 30
# min.node.size = 4 
# mtry = 5
# num.trees = 8 
# splitrule = "gini"
# -----------------------
# Smaller Sample size
# max.depth = 42
# min.node.size = 7
# mtry = 6
# num.trees = 1708
# splitrule = "gini"

# Visualize the tuning results
autoplot(instance_RF)

Tuning_RF_numtrees <- patchwork::wrap_plots(autoplot(instance_RF, type = "marginal", cols_x = "x_domain_num.trees"))
Tuning_kknn_minnodesize <- patchwork::wrap_plots(autoplot(instance_RF, type = "marginal", cols_x = "x_domain_min.node.size"))
Tuning_kknn_mtry <- patchwork::wrap_plots(autoplot(instance_RF, type = "marginal", cols_x = "x_domain_mtry"))
Tuning_kknn_splitrule <- patchwork::wrap_plots(autoplot(instance_RF, type = "marginal", cols_x = "x_domain_splitrule"))

Tuning_RF_numtrees <- Tuning_RF_numtrees +
  labs(x = "number trees", y = "Accuracy") +
  theme_bw() +
  theme(legend.position = "none")

Tuning_kknn_minnodesize <- Tuning_kknn_minnodesize +
  labs(x = "min node size", y = "Accuracy") +
  theme_bw() +
  scale_x_continuous(breaks = seq(1, 11, by = 2)) +
  theme(legend.position = "none")

Tuning_kknn_mtry <- Tuning_kknn_mtry +
  labs(x = "mtry", y = "Accuracy") +
  theme_bw() +
  theme(legend.position = "none") +
  scale_x_continuous(breaks = seq(2, 26, by = 3))

Tuning_kknn_splitrule <- Tuning_kknn_splitrule +
  labs(x = "splitrule", y = "Accuracy") +
  theme_bw() 

# Combine the plots
Tuning_RF_combined <- Tuning_kknn_minnodesize + Tuning_RF_numtrees +
  plot_layout(nrow = 1) +
  plot_annotation(title = "Random Forest Hyperparameter Tuning Results")

ggsave("Tuning_RF_30_NextStep4.png", plot = Tuning_RF_combined, width = 8, height = 5, dpi = 800)
# -------------------------- -------------------------------------------------------------------------------------------------
# Task:
task_RF = as_task_classif(Data, target = "status_group")
task_RF$set_col_roles("status_group", c("target", "stratum"))
# Learner:
learner_RF_tuned = lrn("classif.ranger",
  #num.trees  = 1000,
  #mtry = 5,
  #min.node.size = 2,
  num.threads = 8,
  #splitrule = "gini",
  importance = "impurity"
)

learner_RF_tuned$train(task_RF)
predictions = learner_RF_tuned$predict_newdata(Test_Data)
table(prediction$response)

#learner_RF_tuned$train(task_RF)
#prediction = learner_RF_tuned$predict_newdata(Test_Data)
#table(prediction$response)

# Train the tuned learner
graph_learner_hyp = as_learner(imp_sel %>>% po(lrn("classif.ranger",  
                                                   #num.trees  = 1000,
                                                   #mtry = 6,
                                                   #min.node.size = 1,
                                                   num.threads = 8,
                                                   #splitrule = "gini",
                                                   importance = "impurity")))
graph_learner_hyp$train(task_RF)
#---------------------------------------------------------------------------------------------------------------------------
# Predict on the test set
prediction = graph_learner_hyp$predict_newdata(Test_Data)

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "RF_withNAs.csv", row.names = FALSE)
# -------------------------- -------------------------------------------------------------------------------------------------
# Importance of the features
importance_scores = graph_learner_hyp$importance()
importance_dt = data.table(
    Feature = names(importance_scores),
    Importance = as.numeric(importance_scores)
  )

names <- c("quantity","longitude","latitude","construction year","extraction type 3","payment intervals","GPS height","waterpoint type",
           "region","extraction type 1","available water","district code","source","waterpoint type 2","population")


# Sort by importance
importance_dt = importance_dt[order(-Importance)] %>% head(15)
importance_dt$Feature = names
# Visualize
plot_importance = ggplot(importance_dt, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() + # Flip coordinates to make it a horizontal bar plot
  labs(x = "Feature",
       y = "Importance") +
  theme_bw()

ggsave("Importance_RF.png", plot = plot_importance, width = 8, height = 5, dpi = 800)
# -------------------------- -------------------------------------------------------------------------------------------------
# KKNN with mlr3, Runtime approximately 
# Define the task
# Parallelization: use 6 cores
future::plan("multisession", workers = 6)
# Create a task with the full dataset
task_KKNN <- as_task_classif(Data, target = "status_group")
# Use Features from the feature selection (see ModelsWithFeatureSelection.r) we do not source the file beacause of the long runtime
task_KKNN$select(c("amount_tsh","construction_year","district_code","extraction_type_class","mean_latitude","mean_longitude",
                   "month_recorded","payment_type","quantity","region","source_type","waterpoint_type","year_recorded"))
# Define the learner and the tuning spaces
learner_KKNN = as_learner(ppl("robustify") %>>% po("scale") %>>% lrn("classif.kknn",
  k = to_tune(p_int(1, 30)),
  distance = to_tune(p_fct(c(1, 2))),
  kernel = to_tune(p_fct(c("triangular", "epanechnikov", "biweight", "inv", "gaussian")))
))
# Define the tuning instance
instance_KKNN = ti(
  task = task_KKNN,
  learner = learner_KKNN,
  resampling = rsmp("cv", folds = 3),
  measures = msr("classif.acc"),
  terminator = trm("combo", list(trm("clock_time", stop_time = Sys.time() + 0.5 * 3600),
                                 trm("evals", n_evals = 100)), any = TRUE)
)
learner_KKNN_2 = as_learner(ppl("robustify") %>>% po("scale") %>>% lrn("classif.kknn",
  k = to_tune(p_int(1, 30)),
  distance = 1,
  kernel = "inv"
))
# Define the tuning instance
instance_KKNN_2 = ti(
  task = task_KKNN,
  learner = learner_KKNN_2,
  resampling = rsmp("cv", folds = 3),
  measures = msr("classif.acc"),
  terminator = trm("combo", list(trm("clock_time", stop_time = Sys.time() + 0.5 * 3600),
                                 trm("evals", n_evals = 40)), any = TRUE)
)
tuner_KKNN = tnr("grid_search", batch_size = 40)
# Tune the hyperparameters
tuner_KKNN$optimize(instance_KKNN)
tuner_KKNN$optimize(instance_KKNN_2)
# Result:
instance_KKNN$result$learner_param_vals
# classif.acc = 0.7881481
# k = 25
# distance = 1
# kernel = "inv"

# Visualize the tuning results
autoplot(instance_KKNN_2)

Tuning_kknn_k <- patchwork::wrap_plots(autoplot(instance_KKNN, type = "marginal", cols_x = "classif.kknn.k"))
Tuning_kknn_distance <- patchwork::wrap_plots(autoplot(instance_KKNN, type = "marginal", cols_x = "classif.kknn.distance"))
Tuning_kknn_kernel <- patchwork::wrap_plots(autoplot(instance_KKNN, type = "marginal", cols_x = "classif.kknn.kernel"))

Tuning_kknn_k <- Tuning_kknn_k +
  labs(x = "k", y = "Accuracy") +
  theme_bw() +
  scale_x_continuous(breaks = seq(0, 30, 5)) +
  theme(legend.position = "none")

Tuning_kknn_distance <- Tuning_kknn_distance +
  labs(x = "Distance", y = "Accuracy") +
  theme_bw() +
  scale_x_discrete(labels = c("1" = "Manhattan", "2" = "Euclidean")) +
  theme(legend.position = "none")

Tuning_kknn_kernel <- Tuning_kknn_kernel +
  labs(x = "Kernel", y = "Accuracy") +
  theme_bw() +
  scale_x_discrete(guide = guide_axis(n.dodge = 2))

# Combine the plots
Tuning_kknn_combined <- Tuning_kknn_k + Tuning_kknn_distance + Tuning_kknn_kernel +
  plot_layout(ncol = 3) +
  plot_annotation(title = "k-Nearest-Neighbors Hyperparameter Tuning Results")

ggsave("Tuning_kknn.png", plot = Tuning_kknn_combined, width = 10, height = 4, dpi = 800)
# -------------------------- -------------------------------------------------------------------------------------------------
# Create new learner with the best hyperparameters
learner_KKNN_tuned = as_learner(ppl("robustify") %>>% po("scale") %>>% lrn("classif.kknn",
  k = 25,
  distance = 1,
  kernel = "inv"
))
# Train the tuned learner
learner_KKNN_tuned$train(task_KKNN)
#---------------------------------------------------------------------------------------------------------------------------
# Predict on the test set
prediction = learner_KKNN_tuned$predict_newdata(Test_Data)

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "TunedKKNN.csv", row.names = FALSE)
# ---------------------------------------------------------------------------------------------------------------------------
# Stacked Model with Ranger and KNN as base learners and Ranger as super learner
# Base Learner 1: Ranger (Random Forest) on numerical features
task_stacked <- as_task_classif(Data, target = "status_group")
#Select features for kknn
Features_kknn <- c("amount_tsh","construction_year","district_code","extraction_type_class","mean_latitude","mean_longitude",
                   "month_recorded","payment_type","quantity","region","source_type","waterpoint_type","year_recorded")

po_ranger_branch = po("learner_cv", lrn("classif.ranger",
                                        num.trees  = 1000,
                                        mtry = 6,
                                        min.node.size = 3,
                                        splitrule = "gini",
                                        num.threads = 6),
                        id = "ranger_base")

numeric_feats = task_stacked$feature_types[type == "numeric", id]
ordinal_feats = c("payment_type")
nominal_feats = setdiff(task_stacked$feature_names, c(numeric_feats, ordinal_feats))

nominal_pipe = po("select", id = "select_nominal", selector = selector_name(nominal_feats)) %>>%
  po("encode", method = "one-hot", id = "encode_nominal")
ordinal_pipe = po("select", id = "select_ordinal", selector = selector_name(ordinal_feats)) %>>%
  po("encode", method = "treatment", id = "encode_ordinal")
numeric_pipe = po("select", id = "select_numeric", selector = selector_name(numeric_feats))

feature_union = gunion(list(nominal_pipe,ordinal_pipe,numeric_pipe)) %>>%
  po("featureunion", id = "combined_features")

po_xgboost_branch = feature_union %>>% po("learner_cv", lrn("classif.xgboost",
                                         max_depth = 60,
                                         eta = 0.019, 
                                         nrounds = 300, 
                                         subsample = 0.6791455,
                                         objective      = "multi:softprob",
                                         nthread = 5),
                        id = "xgboost_base")

po_kknn_branch = po("select", selector = selector_name(Features_kknn)) %>>%
                  ppl("robustify") %>>% po("scale") %>>% po("learner_cv", lrn("classif.kknn",
                                                                            k = 25,
                                                                            distance = 1,
                                                                            kernel = "inv"),
                    id = "kknn_base")

base_learner_predictions_graph = gunion(list(po_ranger_branch, po_kknn_branch, po_xgboost_branch)) %>>%
                          po("featureunion")

super_learner_model = lrn("classif.ranger", id = "super.ranger", num.threads = 6)

stacked_graph = base_learner_predictions_graph %>>%
            po("learner", learner = super_learner_model, id = "meta_model")

stacked_learner = as_learner(stacked_graph)
# Train the stacked learner
stacked_learner$train(task_stacked)
#---------------------------------------------------------------------------------------------------------------------------
# Predict on the test set
prediction = stacked_learner$predict_newdata(Test_Data)

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "TunedStacked.csv", row.names = FALSE)
# ---------------------------------------------------------------------------------------------------------------------------

