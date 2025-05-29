# Grid search for the best hyperparameters with caret
# Load necessary packages
library(dplyr)
library(ranger)
library(caret)
library(ggplot2)
library(doParallel)

# Source R file for training data
here::here()
source("CreateTrainingData.R")
# Load the training data
Data <- TrainingData_factor_ordered

# Set seed for reproducibility
set.seed(24)

# Create a data partition for Hyperparameter training
n <- nrow(Data)
Data_hyperparameter <- sample_n(Data, size = n * 0.2) # Sample a subset for hyperparameter tuning
Data_hyperparameter$status_group <- factor(Data_hyperparameter$status_group, levels = c("functional", "non functional", "functional needs repair"), labels = c("functional", "functional_needs_repair", "non_functional"))
Data_hyperparameter <- Data_hyperparameter %>%
  select(-installer) # Remove installer, too many levels

# creatw Data Set for final training
Data_final_Training <- Data %>%
  mutate(status_group = factor(status_group, 
                             levels = c("functional", "functional needs repair", "non functional"), 
                             labels = c("functional", "functional_needs_repair", "non_functional"))) %>%
  select(-installer) # Remove installer for final training

# Hyperparameter tuning:
# Create a grid of hyperparameters to tune

tune_grid <- expand.grid(
  .mtry = c(1, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16), # Values for mtry
  .splitrule =  c("gini", "extratrees"), # Values for splitrule
  .min.node.size = c(1, 3, 5) # Values for min.node.size
)

# Specify resampling strategy
train_control <- trainControl(
  method = "repeatedcv",        # Cross-validation
  number = 10,                  # 10 folds            
  repeats = 3,                  # 3 repeats
  verboseIter = TRUE,           # Show progress
  summaryFunction = multiClassSummary
)

# Parrallel processing setup
num_cores <- detectCores() / 2
cl <- makeCluster(num_cores)

# Start parallel processing
registerDoParallel(cl)

# Train the model with hyperparameter tuning
# Longt runtime, Results are shown below
tuned_model <- train(
  status_group ~ .,
  data = Data_hyperparameter,
  method = "ranger",
  tuneGrid = tune_grid,
  trControl = train_control,
  metric = "Kappa", # Use Kappa for multi-class classification
  importance = "impurity", # To get feature importance later
  num.trees = 500, # Fixed number of trees for this tuning
  respect.unordered.factors = TRUE, # Respect unordered factors
  seed = 24 # For reproducibility of random processes in ranger
)

# Stop parallel processing
stopCluster(cl)

# View the results
print(tuned_model)
plot(tuned_model)


best_params <- tuned_model$bestTune
best_params
# Best hyperparameters
#    mtry splitrule min.node.size
#      14      gini             1

# Train Model with Best Parameters
# Train on the full dataset with the best parameters
model_best_parameter_RandomForest <- ranger(
  formula = status_group ~ ., 
  data = Data_final_Training,
  num.trees = 1000,
  mtry = 14,
  splitrule = "gini",
  min.node.size = 1,
  respect.unordered.factors = TRUE,
  importance = "permutation",
  seed = 24,
  num.threads = num_cores # Specify the number of threads here
  )

model_best_parameter_RandomForest$prediction.error

# Variable importance
importance_scores <- model_best_parameter_RandomForest$variable.importance
sorted_importance <- sort(importance_scores, decreasing = TRUE)
sorted_importance

# Visualize variable importance
importance_df <- data.frame(
  Variable = names(sorted_importance),
  Importance = sorted_importance
  )

importance_Variables <- ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance", x = "Variable", y = "Importance")

importance_Variables

# Predict on the test data
source("CreateTestDataSet.R")
testing_set <- TestData_factor_ordered %>%
  select(-installer)# Remove installer for testing
id <- Id

pred <- predict(model_best_parameter_RandomForest, data = testing_set)

# Save the predictions to a CSV file
submission <- data.frame(id = id, status_group = pred$predictions)
submission$status_group <- gsub("_", " ", submission$status_group)
write.csv(submission, file = "submissionRandomForest_withoutInstaller.csv", row.names = FALSE)
