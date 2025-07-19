# Random Forest with mlr3

# Load required packages
library(mlr3)
library(mlr3learners) # Provides learners like 'classif.ranger'
library(mlr3filters) # For feature selection
library(mlr3viz)     # For plotting
library(mlr3tuning)  # For hyperparameter tuning
library(mlr3fselect) # For feature selection
library(ranger)      # The underlying Random Forest implementation
library(iml)         # For model interpretation (including SHAP)
library(data.table)  # mlr3 often works with data.table
library(mlr3pipelines) # For pipelines

# Source R file for training data
here::here()
source("CreateTrainingData.R")
# Load the training data
Data <- TrainingData_FormattedInstaller %>%
    select(-extraction_type)
# Set seed for reproducibility
set.seed(24)
# Create a classification task
task <- as_task_classif(Data, target = "status_group")
print(task)
# Create training/test split
train_data <- sample(task$row_ids, 0.8 * task$nrow)
test_data <- setdiff(task$row_ids, train_data)
# Create a learner for classification using rpart
learner = lrn("classif.ranger", importance = "impurity") # Using impurity for feature importance
# Train the learner
learner$train(task, row_ids = train_data)
# View the model
learner$model
# Predict on the train and test set
pred_train = learner$predict(task, row_ids=train_data)
pred_test = learner$predict(task, row_ids=test_data)

pred_train$confusion
# Calculate accuracy on the training set
measures = msrs(c('classif.acc'))
pred_train$score(measures)

pred_test$confusion
# Calculate accuracy on the test set
pred_test$score(measures)

# Feature importance using impurity
variab_filter = flt("importance", learner = learner)
variab_filter$calculate(task)
head(as.data.table(variab_filter), 10)

# Feature slelection
tsk_fsel = as_task_classif(Data, target = "status_group")
tsk_fsel$select(c("amount_tsh","gps_height","longitude","latitude","region","population","construction_year",
                  "extraction_type","management_group","payment","quality_group","quantity","source_type","source_class",
                  "waterpoint_type","installer","extraction_type_class"))

instance = fselect(
  fselector = fs("sequential"),
  task =  tsk_fsel,
  learner =   learner,
  resampling = rsmp("cv", folds = 3),
  measure = msr("classif.acc")
)

instance$result_feature_set
# Result:
#[1] "amount_tsh"        "construction_year" "extraction_type"
#[4] "gps_height"        "installer"         "latitude"
#[7] "longitude"         "management_group"  "payment"
#[10] "population"        "quality_group"     "quantity"
#[13] "region"            "source_class"      "source_type"
#[16] "waterpoint_type"

tsk_pen = as_task_classif(Data, target = "status_group")
tsk_pen$select(instance$result_feature_set)

learner$train(task_pen, row_ids = train_data)

pred_test_pen = learner$predict(task, row_ids=test_data)
pred_test_pen$score(measures)

learner = lrn("classif.ranger",
  num.trees  = to_tune(c(500, 2000)),
  mtry = to_tune(2:17),
  min.node.size = to_tune(1:10),
  splitrule = to_tune(c("gini", "extratrees")),
  num.threads = 6
)

instance = ti(
  task = tsk_pen,
  learner = learner,
  resampling = rsmp("cv", folds = 3),
  measures = msr("classif.ce"),
  terminator = trm("run_time", secs = 1800) 
)

tuner = tnr("random_search")
# Tune the hyperparameters
tuner$optimize(instance)

# Result: 
#min.node.size   mtry num.trees splitrule
#           9      6      1000      gini

learner_final = lrn("classif.ranger",
  num.trees  = 1000,
  mtry = 6,
  min.node.size = 9,
  splitrule = "gini",
  num.threads = 6
)

learner_final$train(tsk_pen)
print(learner_final$model)


# --------------------------------------------------------------------------------------------------------------------------
# Load the test data
source("CreateTestDataSet.R")
ID <- Id
Test_Data <- TestData_FormattedInstaller %>%
    select(-extraction_type)

prediction = learner_final$predict_newdata(Test_Data)
prediction

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "submissionRandomForest_mlr3.csv", row.names = FALSE)
# -------------------------- -------------------------------------------------------------------------------------------------



# Best Data Set
BestData <- readr::read_csv("submissionRandomForest_FormattedInstallerII.csv")
table(BestData$status_group)

unique(Data$extraction_type)
unique(Test_Data$extraction_type)
unique(Data$extraction_type_class)
unique(Test_Data$extraction_type_class)

task <- TaskClassif$new(id = "Well_task", backend = Data, target = "status_group")
print(task)

















# Create a ranger learner
# Set importance = "permutation" or "impurity" for global importance
learner_rf_global <- lrn("classif.ranger", importance = "permutation")

# Train the learner
learner_rf_global$train(task)

# Get global feature importance
global_importance <- learner_rf_global$importance()
print(global_importance)

# Visualize global importance (using mlr3viz)
autoplot(learner_rf_global) # This will often show importance if available


importance_df <- as.data.frame(global_importance)
colnames(importance_df) <- "Importance"
importance_df$Feature <- rownames(importance_df)

library(ggplot2)
ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col() +
  coord_flip() + # Makes the bars horizontal
  labs(title = "Feature Importance", x = "Feature", y = "Permutation Importance") +
  theme_minimal()


# Create a ranger learner with local.importance = TRUE
learner_rf_local <- lrn(
  "classif.ranger",
  importance = "permutation",    # Required for local.importance
  scale.permutation.importance = TRUE        # Enable local importance calculation
)

# Train the learner
learner_rf_local$train(task)

# Access the local importance matrix
# This is stored in the 'model' object of the learner
local_imp_matrix <- learner_rf_local$model$variable.importance.local
head(local_imp_matrix)
dim(local_imp_matrix) # N observations x N features



# New Task for variable Selection
source("CreateTrainingData.R")
Data_VarSel <- TrainingData_VarSel %>% sample_frac(0.2)
Data_Eval <- TrainingData_VarSel

set.seed(42)
task_VarSel <- as_task_classif(Data_VarSel, target = "status_group")

learner_VarSel <- lrn("classif.ranger")

instance = fselect(
  fselector = fs("sequential"),
  task =  task_VarSel,
  learner = learner_VarSel,
  resampling = rsmp("cv", folds = 3),
  measure = msr("classif.acc")
)

instance$result_feature_set

# Get the names of the features that are not in the result feature set
setdiff(names(Data_VarSel), instance$result_feature_set)


taskEval <- as_task_classif(Data_Eval, target = "status_group")
#taskEval$select(instance$result_feature_set)
# Create training/test split
train_data <- sample(taskEval$row_ids, 0.8 * taskEval$nrow)
test_data <- setdiff(taskEval$row_ids, train_data)
# Create a learner for classification using rpart
learner = lrn("classif.ranger", importance = "impurity") # Using impurity for feature importance
# Train the learner
learner$train(taskEval, row_ids = train_data)
# View the model
learner$model
# Predict on the test set
pred_test = learner$predict(taskEval, row_ids=test_data)
# Calculate accuracy on the test set
measures = msrs(c('classif.acc'))
pred_test$confusion
# Calculate accuracy on the test set
pred_test$score(measures)

# Feature importance using impurity
variab_filter = flt("importance", learner = learner)
variab_filter$calculate(taskEval)
as.data.table(variab_filter)

# Tuning the hyperparameters
Task_tuning = as_task_classif(Data_VarSel, target = "status_group")

learner = lrn("classif.ranger",
  num.trees  = to_tune(c(500, 1000, 1500)),
  mtry = to_tune(2:15),
  min.node.size = to_tune(1:10),
  splitrule = to_tune(c("gini", "extratrees")),
  num.threads = 6
)

instance_VarSel = ti(
  task = Task_tuning,
  learner = learner,
  resampling = rsmp("cv", folds = 3),
  measures = msr("classif.ce"),
  terminator = trm("none") 
)

tuner = tnr("random_search")
# Tune the hyperparameters
tuner$optimize(instance_VarSel)
autoplot(instance, type = "surface")

str(Data_Eval)
ncol(Data_Eval)


ncol(Data_VarSel)
instance$result_feature_set
names(Data_VarSel)
