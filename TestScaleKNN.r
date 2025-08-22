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

# Source R file for training data
here::here()
# --------------------------------------------------------------------------------------------------------------------------
# Load the training data
source("CreateTrainingData.R")
Data <- TrainingData_Formatted
# --------------------------------------------------------------------------------------------------------------------------
empty_matrix <- matrix(NA, nrow = 10, ncol = 2)
Result <- data.frame(empty_matrix)

Features_kknn <- c("amount_tsh","construction_year","district_code","extraction_type_class","mean_latitude","mean_longitude",
                   "month_recorded","payment_type","quantity","region","source_type","waterpoint_type","year_recorded")


for(i in 1:10) {
    set.seed(i)
    task_1 <- as_task_classif(Data, target = "status_group")
    task_2 <- as_task_classif(Data, target = "status_group")
    
    learner_1 = lrn("classif.ranger", num.trees  = 1500, mtry = 8, min.node.size = 5, splitrule = "gini", num.threads = 8)
    learner_2 = lrn("classif.ranger", num.trees  = 1500, mtry = 8, min.node.size = 3, splitrule = "gini", num.threads = 8)

    train_data <- sample(task_1$row_ids, 0.8 * task_1$nrow)
    test_data <- setdiff(task_1$row_ids, train_data)

    learner_1$train(task_1, row_ids = train_data)
    learner_2$train(task_2, row_ids = train_data)

    prediction_1 = learner_1$predict(task_1, row_ids = test_data)
    prediction_2 = learner_2$predict(task_2, row_ids = test_data)

    measure = msr("classif.acc")

    acc_1 = prediction_1$score(measure)
    acc_2 = prediction_2$score(measure)

    Result[i, 1] <- acc_1
    Result[i, 2] <- acc_2
    print(paste("Iteration:", i, "Accuracy without scaling:", acc_1, "Accuracy with scaling:", acc_2))
}

mean(Result[, 1], na.rm = TRUE)
mean(Result[, 2], na.rm = TRUE)













# Nested CV 
future::plan("multisession", workers = 3)
ps <- makeParamSet(
  makeIntegerParam("mtry", lower = 2, upper = 26),
  makeIntegerParam("num.trees", lower = 500, upper = 2000),
  makeIntegerParam("min.node.size", lower = 1, upper = 10)),
  makeDiscreteVectorParam("splitrule", len = 2, values = c("gini", "extratrees")
)
ctrl = makeTuneControlGrid(budget = 5)
inner = makeResampleDesc("CV", iters = 3)

lrn = makeTuneWrapper("classif.ranger", resampling = inner, par.set = ps, control = ctrl, show.info = FALSE)
# Outer resampling loop
outer = makeResampleDesc("CV", iters = 3)
r = resample(lrn, task_RF, resampling = outer, extract = getTuneResult, show.info = FALSE)




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

# -------------------------- -------------------------------------------------------------------------------------------------
# Visualize the Cross-Validation:
mtry <- ggplot(as.data.table(instance_RF_2$archive), aes(x = classif.ranger.mtry, y = classif.acc)) +
  geom_point() +
  geom_smooth(method = "loess") +
  theme_bw() +
  labs(title = "Accuracy vs mtry", y = "Accuracy", x = "mtry")

ggsave("mtry.png", plot = mtry, width = 6, height = 4, dpi = 800)

# -------------------------- -------------------------------------------------------------------------------------------------
# Nested Tuning
task_RF_Nested <- as_task_classif(Evaluation, target = "status_group")
# Define the learner and the tuning spaces
learner_RF_nested = lrn("classif.ranger",
  num.trees  = 1500,
  mtry = to_tune(p_int(4, 12)),
  min.node.size = to_tune(p_int(1, 10)),
  splitrule = "gini",
  num.threads = 4
)

tune_instance_RF <- tune_nested(
  tuner = tnr("grid_search", batch_size = 80),
  task = task_RF_Nested,
  learner = learner_RF_nested,
  inner_resampling = rsmp("cv", folds = 3),
  outer_resampling = rsmp("cv", folds = 3),
  measure = msr("classif.acc"),
  terminator = trm("combo", list(trm("clock_time", stop_time = Sys.time() + 1.5 * 3600),
                                 trm("evals", n_evals = 80)), any = TRUE),
  store_tuning_instance =  TRUE
)

inner_tuning_results_list = extract_inner_tuning_results(tune_instance_RF)
inner_tuning_results_list
#iteration min.node.size  mtry classif.acc learner_param_vals  x_domain      task_id           learner_id resampling_id
#      <int>         <int> <int>       <num>             <list>    <list>       <char>               <char>        <char>
#         1             9     6   0.7940476          <list[5]> <list[2]>    Evaluation classif.ranger.tuned            cv
#         2             9     6   0.7955988          <list[5]> <list[2]>    Evaluation classif.ranger.tuned            cv
#         3             2     4   0.7973665          <list[5]> <list[2]>    Evaluation classif.ranger.tuned            cv
# Median = mtry 6, min.node.size = 9, num.trees = 1500, splitrule = "gini"
# -------------------------- -------------------------------------------------------------------------------------------------
# Create new learner with the best hyperparameters
task_RF <- as_task_classif(Data, target = "status_group")