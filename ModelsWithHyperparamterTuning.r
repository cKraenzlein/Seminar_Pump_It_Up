# Load required packages
library(mlr3)
library(mlr3learners)   # Provides learners like 'classif.ranger'
library(mlr3filters)    # For feature selection
library(mlr3viz)        # For plotting
library(mlr3tuning)     # For hyperparameter tuning
library(mlr3fselect)    # For feature selection
library(ranger)         # The underlying Random Forest implementation
library(iml)            # For model interpretation (including SHAP)
library(data.table)     # mlr3 often works with data.table
library(mlr3pipelines)  # For pipelines
library(ggplot2)        # For plotting
library(dplyr)          # For data manipulation
library(patchwork)      # Combining Plots

# Source R file for training data
here::here()
# --------------------------------------------------------------------------------------------------------------------------
# Load the training data
source("CreateTrainingData.R")
Data <- TrainingData_Formatted
# --------------------------------------------------------------------------------------------------------------------------
# Load the test data
source("CreateTestDataSet.R")
ID <- Id
Test_Data <- TestData_Formatted
#---------------------------------------------------------------------------------------------------------------------------
# Set seed for reproducibility
set.seed(24)
# --------------------------------------------------------------------------------------------------------------------------
# Random Forest with mlr3, Runtime approximately 2h 30min
# Define the task
task_RF <- as_task_classif(Data, target = "status_group")
# Define the learner and the tuning spaces
learner_RF = as_learner(ppl("robustify") %>>% lrn("classif.ranger",
  num.trees  = to_tune(p_int(500, 2000)),
  mtry = to_tune(p_int(2, 26)),
  min.node.size = to_tune(p_int(1, 10)),
  splitrule = to_tune(p_fct(c("gini", "extratrees"))),
  num.threads = 5
))
# Define the tuning instance
instance_RF = ti(
  task = task_RF,
  learner = learner_RF,
  resampling = rsmp("cv", folds = 3),
  measures = msr("classif.acc"),
  terminator = trm("combo", list(trm("clock_time", stop_time = Sys.time() + 3 * 3600),
                                 trm("evals", n_evals = 150)), any = TRUE)
)
tuner_RF = tnr("random_search")
# Tune the hyperparameters
tuner_RF$optimize(instance_RF)
# Result:
instance_RF$result$learner_param_vals
# min.node.size = 8
# mtry = 25
# num.trees = 1500
# splitrule = "gini"

# Visualize the tuning results
autoplot(instance_RF)

Tuning_RF_numtrees <- patchwork::wrap_plots(autoplot(instance_RF, type = "marginal", cols_x = "classif.ranger.num.trees"))
Tuning_kknn_minnodesize <- patchwork::wrap_plots(autoplot(instance_RF, type = "marginal", cols_x = "classif.ranger.min.node.size"))
Tuning_kknn_mtry <- patchwork::wrap_plots(autoplot(instance_RF, type = "marginal", cols_x = "classif.ranger.mtry"))
Tuning_kknn_splitrule <- patchwork::wrap_plots(autoplot(instance_RF, type = "marginal", cols_x = "classif.ranger.splitrule"))

Tuning_RF_numtrees <- Tuning_RF_numtrees +
  labs(x = "number trees", y = "Accuracy") +
  theme_bw() +
  theme(legend.position = "none")

Tuning_kknn_minnodesize <- Tuning_kknn_minnodesize +
  labs(x = "min node size", y = "Accuracy") +
  theme_bw() +
  scale_x_continuous(breaks = seq(1, 10, by = 2)) +
  theme(legend.position = "none")

Tuning_kknn_mtry <- Tuning_kknn_mtry +
  labs(x = "mtry", y = "Accuracy") +
  theme_bw() +
  theme(legend.position = "none")

Tuning_kknn_splitrule <- Tuning_kknn_splitrule +
  labs(x = "splitrule", y = "Accuracy") +
  theme_bw() 

# Combine the plots
Tuning_RF_combined <- Tuning_kknn_minnodesize + Tuning_kknn_mtry + Tuning_RF_numtrees + Tuning_kknn_splitrule +
  plot_layout(ncol = 2, nrow = 2) +
  plot_annotation(title = "Random Forest Hyperparameter Tuning Results")

ggsave("Tuning_RF.png", plot = Tuning_RF_combined, width = 10, height = 8, dpi = 800)
# -------------------------- -------------------------------------------------------------------------------------------------
# Second Tuning Grid 
# Define the learner and the tuning spaces
learner_RF_2 = as_learner(ppl("robustify") %>>% lrn("classif.ranger",
  num.trees  = to_tune(p_int(500, 2000)),
  mtry = to_tune(p_int(15, 26)),
  min.node.size = to_tune(p_int(1, 10)),
  splitrule = "gini",
  num.threads = 8
))
# Define the tuning instance
instance_RF_2 = ti(
  task = task_RF,
  learner = learner_RF_2,
  resampling = rsmp("cv", folds = 5),
  measures = msr("classif.acc"),
  terminator = trm("combo", list(trm("clock_time", stop_time = Sys.time() + 2.5 * 3600),
                                 trm("evals", n_evals = 100)), any = TRUE)
)
# Tune the hyperparameters
tuner_RF$optimize(instance_RF_2)
# Result:
instance_RF_2$result$learner_param_vals
# num.trees = 1432, mtry = 26, min.node.size = 5

Tuning_RF_numtrees_2 <- patchwork::wrap_plots(autoplot(instance_RF_2, type = "marginal", cols_x = "classif.ranger.num.trees"))
Tuning_kknn_minnodesize_2 <- patchwork::wrap_plots(autoplot(instance_RF_2, type = "marginal", cols_x = "classif.ranger.min.node.size"))
Tuning_kknn_mtry_2 <- patchwork::wrap_plots(autoplot(instance_RF_2, type = "marginal", cols_x = "classif.ranger.mtry"))


Tuning_RF_numtrees_2 <- Tuning_RF_numtrees_2 +
  labs(x = "number trees", y = "Accuracy") +
  theme_bw() 

Tuning_kknn_minnodesize_2 <- Tuning_kknn_minnodesize_2 +
  labs(x = "min node size", y = "Accuracy") +
  theme_bw() +
  scale_x_continuous(breaks = seq(1, 10, by = 2)) +
  theme(legend.position = "none")

Tuning_kknn_mtry_2 <- Tuning_kknn_mtry_2 +
  labs(x = "mtry", y = "Accuracy") +
  theme_bw() +
  theme(legend.position = "none") +
  scale_x_continuous(breaks = seq(15, 26, by = 2))

# Combine the plots
Tuning_RF_combined_2 <- Tuning_kknn_minnodesize_2 + Tuning_kknn_mtry_2 + Tuning_RF_numtrees_2 + 
  plot_layout(ncol = 3) +
  plot_annotation(title = "Random Forest Hyperparameter Tuning Results", 
                  subtitle = "Performance metric = accuracy, Resampling strategy = CV 5-folds, Random search")

ggsave("Tuning_RF_2_sub.png", plot = Tuning_RF_combined_2, width = 10, height = 4, dpi = 800)

# Visualize the Cross-Validation:
mtry <- ggplot(as.data.table(instance_RF_2$archive), aes(x = classif.ranger.mtry, y = classif.acc)) +
  geom_point() +
  geom_smooth(method = "loess") +
  theme_bw() +
  labs(title = "Accuracy vs mtry", y = "Accuracy", x = "mtry")

ggsave("mtry.png", plot = mtry, width = 6, height = 4, dpi = 800)
# -------------------------- -------------------------------------------------------------------------------------------------
# Create new learner with the best hyperparameters
learner_RF_tuned = lrn("classif.ranger",
  num.trees  = 1500,
  mtry = 26,
  min.node.size = 7,
  splitrule = "gini",
  num.threads = 6,
  importance = "permutation"
)
# Train the tuned learner
learner_RF_tuned$train(task_RF)
#---------------------------------------------------------------------------------------------------------------------------
# Predict on the test set
prediction = learner_RF_tuned$predict_newdata(Test_Data)

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "TunedRFtotal.csv", row.names = FALSE)
# -------------------------- -------------------------------------------------------------------------------------------------
# Importance of the features
importance_scores = learner_RF_tuned$importance()
importance_dt = data.table(
    Feature = names(importance_scores),
    Importance = as.numeric(importance_scores)
  )

names <- c("Menge", "Längengrad", "Breitengrad",
          "Baujahr", "Zahlungsintervall", "Region",
          "GPS Höhe", "Antrieb der Pumpe 1", "Art der Förderung 2",
          "Quelltypen 1", "Art der Förderung 1", "Distrikt Code",
          "Bevölkerungzahl", "Antrieb der Pumpe 2", "Fördermenge",
          "Becken", "Genehmigung", "Verwaltungsart",
          "Quelltypen 2", "Monat erfasst", "Installateur 1",
          "Quelltypen 3", "Jahr erfasst", "Verwaltungsart 2",
          "Öffentliche Treffen", "Qualität")

importance_dt$Feature = names
# Sort by importance
importance_dt = importance_dt[order(-Importance)]
# Visualize
plot_importance = ggplot(importance_dt, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() + # Flip coordinates to make it a horizontal bar plot
  labs(title = "Random Forest Feature Importance",
        x = "Feature",
        y = "Permutation Importance") +
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
                                 trm("evals", n_evals = 100)), any = TRUE)
)
tuner_KKNN = tnr("random_search")
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
                                        num.threads = 6)
                        id = "ranger_base")

po_kknn_branch = po("select", selector = selector_name(Features_kknn)) %>>%
                  ppl("robustify") %>>% po("scale") %>>% po("learner_cv", lrn("classif.kknn",
                                                                            k = 25,
                                                                            distance = 1,
                                                                            kernel = "inv"),
                    id = "kknn_base")

base_learner_predictions_graph = gunion(list(po_ranger_branch, po_kknn_branch)) %>>%
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

