# Load necessary packages
library(dplyr)
library(ipred)
library(randomForest)
library(caret)
library(rpart)
library(doParallel) 
library(class)
library(ranger)

# Set seed for reproducibility
set.seed(24) 

#Load the train data
here::here()
source("CreateTrainingData.R")
Data <- TrainingData_factor_ordered

# Split data into training and testing sets
train_index <- createDataPartition(Data$status_group, p = 0.7, list = FALSE)
train_data <- Data[train_index, ]
test_data <- Data[-train_index, ]

Data_final_Training <- Data %>%
  mutate(status_group = factor(status_group, 
                             levels = c("functional", "functional needs repair", "non functional"), 
                             labels = c("functional", "functional_needs_repair", "non_functional"))) %>%
  select(-installer) # Remove installer for final training


#Create a data partition for training
n <- nrow(Data)
Data_hyperparameter <- sample_n(Data, size = n * 0.2) # Sample a subset for hyperparameter tuning
Data_hyperparameter$status_group <- factor(Data_hyperparameter$status_group, levels = c("functional", "non functional", "functional needs repair"), labels = c("functional", "functional_needs_repair", "non_functional"))
Data_hyperparameter <- Data_hyperparameter %>%
  select(-installer) # Remove installer for hyperparameter tuning

# Specify resampling strategy
cv <- trainControl(
  method = "repeatedcv", 
  number = 10, 
  repeats = 3,
  summaryFunction = multiClassSummary,
  classProbs = TRUE, # Needed for metrics like ROC
  verboseIter = TRUE # Show progress
)

# Create grid of hyperparameter values
hyper_grid <- expand.grid(k = seq(2, 25, by = 1))

num_cores <- detectCores() / 2
cl <- makeCluster(num_cores)
registerDoParallel(cl)
# Tune a knn model using grid search
knn_fit <- train(
  status_group ~ ., 
  data = Data_hyperparameter, 
  method = "knn", 
  trControl = cv, 
  tuneGrid = hyper_grid,
  metric = "Kappa" # Use Kappa for multi-class classification
)
stopCluster(cl)

knn_fit

ggplot(knn_fit)


png("KNN_Fit_Kappa_2_25.png",    # The name of the file
    width = 1200,                               # Width in pixels
    height = 800,                              # Height in pixels
    res = 100)                                  # Resolution (dots per inch)
ggplot(knn_fit)
dev.off()

# Define a tuneGrid with only the best k
final_tune_grid <- expand.grid(k = 5)

# Train control for fitting the final model (no resampling)
final_train_control <- trainControl(
  method = "none" # No resampling
)


# Train the KNN model with the best k = 39
knn_model <- train(
  form = status_group ~ .,
  data = Data_final_Training,
  method = "knn",
  trControl = final_train_control,
  tuneGrid = final_tune_grid
)

source("CreateTestDataSet.R")
testing_set <- TestData_factor_ordered %>%
  select(-installer)# Remove installer for testing
id <- Id

pred <- predict(knn_model, newdata = testing_set)

# Save the predictions to a CSV file
submission <- data.frame(id = id, status_group = pred)
submission$status_group <- gsub("_", " ", submission$status_group)
write.csv(submission, file = "submissionKNN(k=5).csv", row.names = FALSE)










# TEST
tune_grid <- expand.grid(
  .mtry = c(1, 2, 4, 6, 8, 10, 12, 14, 16), # Values for mtry
  .splitrule = c("gini", "extratrees"), # Values for splitrule
  .min.node.size = c(1, 3, 5) # Values for min.node.size
)

train_control <- trainControl(
  method = "repeatedcv",                # Cross-validation
  number = 10,
  repeats = 3,                   # 5 folds
  verboseIter = TRUE,           # Show progress
  classProbs = TRUE,            # Needed for ROC curve if desired
  summaryFunction = multiClassSummary # Use multiClassSummary for > 2 classes
)

num_cores <- detectCores() / 2
cl <- makeCluster(num_cores)
registerDoParallel(cl)
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
stopCluster(cl)

# View the results
print(tuned_model)

# Plot the tuning results
png("GridSearch_withoutInstaller.png",                  # The name of the file
    width = 1200,                               # Width in pixels
    height = 800,                               # Height in pixels
    res = 100)                                  # Resolution (dots per inch)
plot(tuned_model)
dev.off()

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

# Use ggplot2 for a more sophisticated plot (optional)
library(ggplot2)
importance_df <- data.frame(
  Variable = names(sorted_importance),
  Importance = sorted_importance
)

importance_Variables <- ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance", x = "Variable", y = "Importance")
importance_Variables



source("CreateTestDataSet.R")
testing_set <- TestData_factor_ordered %>%
  select(-installer)# Remove installer for testing
id <- Id

pred <- predict(model_best_parameter_RandomForest, data = testing_set)

# Save the predictions to a CSV file
submission <- data.frame(id = id, status_group = pred$predictions)
submission$status_group <- gsub("_", " ", submission$status_group)
write.csv(submission, file = "submissionRandomForest_withoutInstaller.csv", row.names = FALSE)
