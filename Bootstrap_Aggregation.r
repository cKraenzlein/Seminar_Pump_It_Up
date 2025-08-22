# Load necessary packages
library(dplyr)
library(ipred)
library(randomForest)
library(caret)
library(rpart)

#Load the train data
here::here()
source("CreateTrainingData.R")
Data <- TrainingData_factor_ordered

# Split data into training and testing sets
train_index <- createDataPartition(Data$status_group, p = 0.7, list = FALSE)
train_data <- Data[train_index, ]
test_data <- Data[-train_index, ]

# Set seed for reproducibility
set.seed(24)

# Train the bagging model
bagged_model<- bagging(status_group ~ .,
                              data = train_data,
                              nbagg = 1000, # Number of trees
                              coob = TRUE, # Compute out-of-bag error
                              control = rpart.control(minsplit = 2, cp = 0)) # Control complexity of base trees

# Print the model summary
print(bagged_model)

# Make predictions on the test set
predictions_ipred <- predict(bagged_model, newdata = test_data)

# Evaluate performance (e.g., Confusion Matrix)
confusionMatrix <- confusionMatrix(predictions, test_data$status_group)
print(confusionMatrix)





# Hyperparameter tuning for bagging model
#Create a data partition for training
n <- nrow(Data)
Data_hyperparameter <- sample_n(Data, size = n * 0.2) # Sample a subset for hyperparameter tuning
Data_hyperparameter$status_group <- factor(Data_hyperparameter$status_group, levels = c("functional", "non functional", "functional needs repair"), labels = c("functional", "functional_needs_repair", "non_functional"))

# Define training control for cross-validation
# method = "cv": k-fold cross-validation
# number = 10: 10 folds
# repeats = 3: repeat 10-fold CV 3 times
# classProbs = TRUE: Needed for metrics like ROC
# summaryFunction = defaultSummary: Uses Accuracy and Kappa for classification
fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = defaultSummary,
  verboseIter = TRUE # See progress
)

# Define the tuning grid (what parameters to try)
# For "treebag", caret only tunes the number of bootstrap iterations (nbagg).
# You can customize rpart control via the 'control' argument if using 'bagging' directly,
# but 'caret's 'treebag' only exposes 'nbagg' for direct tuning.
# If you want to tune rpart parameters, you'd typically tune rpart first or use a custom model.
tuneGrid_bag <- expand.grid(
  nbagg = c(50, 100, 150, 200) # Number of trees
)

# Train the bagging model using caret
# method = "treebag" uses ipred::bagging
bag_model_caret <- train(status_group ~ .,
                         data = Data_hyperparameter,
                         method = "treebag",
                         trControl = fitControl,
                         tuneGrid = tuneGrid_bag,
                         metric = "F1_Macro" # Metric to optimize
)

# Print the tuning results
print(bag_model_caret)

# Plot the tuning results
plot(bag_model_caret)

# Get the best model
best_bag_model <- bag_model_caret$finalModel
print(best_bag_model)




# make bootstrapping reproducible
set.seed(123)

# Time the training process
start_time <- Sys.time()

# train bagged model
Model_bag1 <- bagging(
  formula = status_group ~ .,
  data = Data_hyperparameter,
  nbagg = 25,  
  coob = TRUE
)

# Time the training process
end_time <- Sys.time()
cat("Elapsed time (manual calculation):", difftime(end_time, start_time, units = "secs"), "seconds\n")


Model_bag1

library(ranger)
# Bagging with ranger for classification

num_cores <- detectCores() / 2

model_ranger <- ranger(
  formula = status_group ~ ., 
  data = Data_hyperparameter,
  num.trees = 200,
  mtry = ncol(Data_hyperparameter) - 1,
  importance = "permutation",
  seed = 24,
  num.threads = num_cores # Specify the number of threads here
  )

model_ranger 



# -----------------------------------------------------------------------------------------
# Hyperparameter tuning for ranger bagging model

# Load necessary libraries
library(doParallel)    # Für Parallelisierung

num_cores <- detectCores() / 2
cl <- makeCluster(num_cores)
registerDoParallel(cl)

tune_grid <- expand.grid(
  .mtry = c(1, 2, 6, 10, 14, 17), # Values for mtry
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

tuned_model <- train(
  status_group ~ .,
  data = Data_hyperparameter,
  method = "ranger",
  tuneGrid = tune_grid,
  trControl = train_control,
  metric = "Kappa", # Use Kappa for multi-class classification
  importance = "impurity", # To get feature importance later,
  num.trees = 500, # Fixed number of trees for this tuning
  respect.unordered.factors = TRUE, # Respect unordered factors
  seed = 24 # For reproducibility of random processes in ranger
)

stopCluster(cl)

# 4. Ergebnisse anzeigen
print(tuned_model)

png("ParameterGridSearchExtended.png",    # The name of the file
    width = 1200,                               # Width in pixels
    height = 800,                              # Height in pixels
    res = 100)                                  # Resolution (dots per inch)
plot(tuned_model)
dev.off()
# Best hyperparameters
#    mtry splitrule min.node.size
#      17      gini             1

# Train on the full dataset with the best parameters
model_best_parameter <- ranger(
  formula = status_group ~ ., 
  data = Data,
  num.trees = 1500,
  mtry = 17,
  splitrule = "gini",
  min.node.size = 1,
  respect.unordered.factors = TRUE,
  importance = "permutation",
  seed = 24,
  num.threads = num_cores # Specify the number of threads here
  )

model_best_parameter$prediction.error

source("CreateTestDataSet.R")
# Load the test data
TestData <- TestData_factor_ordered
id <- Id
pred <- predict(model_best_parameter, data = TestData, type = "response")

# Save the predictions to a CSV file
submission <- data.frame(id = id, status_group = pred$predictions)
# Save the submission file
write.csv(submission, file = "submissionBaggingAggregation.csv", row.names = FALSE)
