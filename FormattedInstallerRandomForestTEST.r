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
Data <- TrainingData_FormattedInstaller

# Set seed for reproducibility
set.seed(24)

# Create a data partition for Hyperparameter training
n <- nrow(Data)
Data_hyperparameter <- sample_n(Data, size = n * 0.15) # Sample a subset for hyperparameter tuning
Data_hyperparameter$status_group <- factor(Data_hyperparameter$status_group, levels = c("functional", "non functional", "functional needs repair"), labels = c("functional", "functional_needs_repair", "non_functional"))

# creatw Data Set for final training
Data_final_Training <- Data %>%
  mutate(status_group = factor(status_group, 
                             levels = c("functional", "functional needs repair", "non functional"), 
                             labels = c("functional", "functional_needs_repair", "non_functional")))

# Hyperparameter tuning:
# Create a grid of hyperparameters to tune

tune_grid <- expand.grid(
  .mtry = c(1, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15), # Values for mtry
  .splitrule =  c("gini"), #c("gini"), #c("gini", "extratrees"), # Values for splitrule
  .min.node.size = c(1) #c(1, 3, 5) # Values for min.node.size
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
  num.trees = 500, # Fixed number of trees for this tuning
  respect.unordered.factors = TRUE, # Respect unordered factors
  seed = 24 # For reproducibility of random processes in ranger
)

# Stop parallel processing
stopCluster(cl)

# View the results
print(tuned_model)
png("HyperparameterRandomForestFormattedInstallerII.png",    # The name of the file
    width = 1200,                               # Width in pixels
    height = 800,                              # Height in pixels
    res = 100)                                  # Resolution (dots per inch)
plot(tuned_model)
dev.off()


best_params <- tuned_model$bestTune
best_params
# Best hyperparameters
#    mtry splitrule min.node.size
#      12      gini             3

# Train Model with Best Parameters
# Train on the full dataset with the best parameters
model_best_parameter_RandomForest <- ranger(
  formula = status_group ~ ., 
  data = Data_final_Training,
  num.trees = 1000,
  mtry = 10,
  splitrule = "gini",
  min.node.size = 1,
  respect.unordered.factors = TRUE,
  importance = "permutation",
  seed = 24,
  num.threads = , # Specify the number of threads here
  local.importance = TRUE # Use local importance for variable importance
  )

model_best_parameter_RandomForest$prediction.error

# Variable importance
importance_scores <- model_best_parameter_RandomForest$variable.importance
sorted_importance <- sort(importance_scores, decreasing = TRUE)
sorted_importance

# Local variable importance
local_imp_matrix <- model_best_parameter_RandomForest$variable.importance.local
head(local_imp_matrix)
dim(local_imp_matrix)

# Combine local importance with the original data
local_imp_df <- as.data.frame(local_imp_matrix)
local_imp_df$status_group <- Data_final_Training$status_group # Add the target variable for grouping

# Calculate average local importance per species for each feature
avg_local_imp_by_target <- local_imp_df %>%
  group_by(status_group) %>%
  summarise(across(everything(), mean, na.rm = TRUE)) # 'everything()' to select all feature columns
print(avg_local_imp_by_target)

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
testing_set <- TestData_FormattedInstallerII
id <- Id

pred <- predict(model_best_parameter_RandomForest, data = testing_set)

# Save the predictions to a CSV file
submission <- data.frame(id = id, status_group = pred$predictions)
submission$status_group <- gsub("_", " ", submission$status_group)
write.csv(submission, file = "submissionRandomForest_FormattedInstallerII.csv", row.names = FALSE)


# TEST 
# Manual loop to test different ntree values with caret
ntree_values <- c(100, 250, 500, 750, 1000, 1500, 2000)
results_ntree <- list()

tune_grid_Numtrees <- expand.grid(
  .mtry = c(12), # Values for mtry
  .splitrule =  c("gini"), # Values for splitrule
  .min.node.size = c(3) # Values for min.node.size
)
# Specify resampling strategy
train_control <- trainControl(
  method = "repeatedcv",        # Cross-validation
  number = 10,                  # 10 folds            
  repeats = 3,                  # 3 repeats
  verboseIter = TRUE,           # Show progress
  summaryFunction = multiClassSummary
)


for (ntree_val in ntree_values) {
  cat("Training with ntree =", ntree_val, "\n")
  model_temp <- train(
    formula = status_group ~ .,
    data = Data_hyperparameter,
    method = "ranger",
    importance = "impurity", # To get feature importance later
    num.trees = 200,
    tuneGrid = tune_grid_Numtrees,
    respect.unordered.factors = TRUE, # Respect unordered factors
    seed = 24, # For reproducibility of random processes in ranger
    trControl = train_control
  )
  results_ntree[[as.character(ntree_val)]] <- model_temp
}

# Extract and plot accuracy for each ntree
accuracy_by_ntree <- sapply(results_ntree, function(x) x$results$Accuracy[which.max(x$results$Accuracy)])
plot(ntree_values, accuracy_by_ntree, type = "b",
     xlab = "Number of Trees", ylab = "Accuracy (CV)",
     main = "Random Forest Accuracy vs. Number of Trees (Caret)")
