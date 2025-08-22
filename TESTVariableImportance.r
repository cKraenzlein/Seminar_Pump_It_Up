# Grid search for the best hyperparameters with caret
# Load necessary packages
library(dplyr)
library(ranger)
library(caret)
library(ggplot2)
library(doParallel)
library(tidyr)

# Source R file for training data
here::here()
source("CreateTrainingData.R")
# Load the training data
Data <- TrainingData

Data <- Data %>%
  dplyr::mutate(across(where(is.character), replace_na, "Unknown")) %>%
  dplyr::mutate(across(where(is.character), as.factor)) %>%
  dplyr::select(-id, -recorded_by, -date_recorded,-wpt_name, -subvillage) %>% # Remove unnecessary columns
  dplyr::select(-quantity_group, -payment_type, -region_code, -num_private) %>% # Doppelt vorhandene Variablen entfernen
  dplyr::select(-permit, -public_meeting, -scheme_management) %>% # Remove variables that conatin NAs and are the importance isnt high
  dplyr::select(-installer, -ward, -scheme_name, -funder) # Remove installer, because it has too many levels 

na_counts <- colSums(is.na(Data))


str(Data)

# Set seed for reproducibility
set.seed(24)

# Create a data partition for Hyperparameter training
n <- nrow(Data)
Data_hyperparameter <- sample_n(Data, size = n * 0.1) # Sample a subset for hyperparameter tuning
Data_hyperparameter$status_group <- factor(Data_hyperparameter$status_group, levels = c("functional", "non functional", "functional needs repair"), labels = c("functional", "functional_needs_repair", "non_functional"))

str(Data_hyperparameter)

# creatw Data Set for final training
Data_final_Training <- Data %>%
  mutate(status_group = factor(status_group, 
                             levels = c("functional", "functional needs repair", "non functional"), 
                             labels = c("functional", "functional_needs_repair", "non_functional")))

# Define the split ratio (e.g., 70% train, 30% test)
train_ratio <- 0.7

# Get the total number of rows
n_rows <- nrow(Data_final_Training)

# Generate random indices for the training set
train_indices <- sample(n_rows, size = floor(train_ratio * n_rows))

# Create the training and test sets
train_df <- Data_final_Training[train_indices, ]
test_df <- Data_final_Training[-train_indices, ]

str(train_df)

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
  repeats = 1,                  # 3 repeats
  verboseIter = TRUE,           # Show progress
  summaryFunction = multiClassSummary
)

# Parrallel processing setup
num_cores <- 7 # Use 7 cores for parallel processing
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
png("HyperparameterRandomForestFormattedInstallerII.png",    # The name of the file
    width = 1200,                               # Width in pixels
    height = 800,                              # Height in pixels
    res = 100)                                  # Resolution (dots per inch)
plot(tuned_model)
dev.off()


best_params <- tuned_model$bestTune
best_params


model_best_parameter_RandomForest <- ranger(
  formula = status_group ~ ., 
  data = train_df,
  num.trees = 1000,
  mtry = 10,
  splitrule = "gini",
  min.node.size = 1,
  respect.unordered.factors = TRUE,
  importance = "permutation",
  seed = 24,
  num.threads = 7 # Specify the number of threads here
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


predicted_labels <- predict(model_best_parameter_RandomForest, data = test_df)$predictions
confusionMatrix(data = predicted_labels, reference = test_df$status_group, mode = "everything")




png("ImportanceAllVariables.png",    # The name of the file
    width = 1200,                               # Width in pixels
    height = 800,                              # Height in pixels
    res = 100)                                  # Resolution (dots per inch)
plot(importance_Variables)
dev.off()

str(Data_final_Training)

source("CreateTestDataSet.R")
TestDataI <- TestData %>%
  mutate(across(where(is.character), replace_na, "Unknown")) %>%
  mutate(across(where(is.character), as.factor)) %>%
  select(-id, -recorded_by, -date_recorded,-wpt_name, -subvillage) %>% # Remove unnecessary columns
  select(-quantity_group, -payment_type, -region_code, -num_private) %>% # Doppelt vorhandene Variablen entfernen
  select(-permit, -public_meeting, -scheme_management) # Remove variables that conatin NAs and are the importance isnt high

id <- Id

pred <- predict(model_best_parameter_RandomForest, data = TestDataI)

# Save the predictions to a CSV file
submission <- data.frame(id = id, status_group = pred$predictions)
submission$status_group <- gsub("_", " ", submission$status_group)
write.csv(submission, file = "submissionRandomForest_AllVariables_NAsReplaced.csv", row.names = FALSE)



na_counts <- colSums(is.na(TestDataI))



library(ggplot2)   # allows extension of visualizations
library(dplyr)     # basic data transformation
library(h2o)       # machine learning modeling
library(iml)       # ML interprtation
library(randomForest)

rf <- randomForest(status_group ~ ., data = train_df, ntree = 1000)

model_best_parameter_RandomForest <- ranger(
  formula = status_group ~ ., 
  data = Data_hyperparameter,
  num.trees = 500,
  mtry = 10,
  splitrule = "gini",
  min.node.size = 1,
  respect.unordered.factors = TRUE,
  importance = "permutation",
  seed = 24,
  num.threads = 7 # Specify the number of threads here
  )


X <- Data_hyperparameter[which(names(Data_hyperparameter) != "status_group")]
predictor <- Predictor$new(model_best_parameter_RandomForest, data = X, y = Data_hyperparameter$status_group)

# Feature importance
imp <- FeatureImp$new(predictor, loss = "")
plot(imp)

effs <- FeatureEffects$new(predictor, grid.size = 10)
plot(effs)
