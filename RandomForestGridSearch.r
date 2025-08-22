# Grid search for the best hyperparameters with caret
# Load necessary packages
library(dplyr)
library(ranger)
library(caret)
library(ggplot2)

# Source R file for training data
here::here()
source("CreateTrainingData.R")
# Load the training data
Data <- TrainingData

# Filter an format the training data
Data_factor_2 <- Data %>% 
    select(status_group, amount_tsh, gps_height, longitude, latitude, region, population, construction_year, extraction_type,
           management_group, payment, quality_group, quantity, source_type, source_class, waterpoint_type, installer, extraction_type_class) %>%
    mutate(status_group = factor(status_group,
                                  levels = c("functional", "functional needs repair", "non functional"),
                                  labels = c("functional", "functional_needs_repair", "non_functional")), 
           waterpoint_type = as.factor(waterpoint_type), # amount of water available
           source_type = as.factor(source_type), # The source of the water, since the variable wasnt that important in the first model, 
                                       # we use source_type because it has less levels
           source_class = as.factor(source_class), # were the water is "stored", groundwater or surface, maybe drop this
           quantity = as.factor(quantity), # quantity of water available / quantity = quantity_group
           quality_group = as.factor(quality_group), # quality of the water, fewer levels than water_quality, in the first model we used water_quality
                                                     # but the variable was not that important                     
           payment = as.factor(payment), # What the water costs / payment_type = payment the level names are different but the variable is the same
           management_group = as.factor(management_group), # How the waterpoint is managed / management_group has less levels than management, not that important in first model
           extraction_type = as.factor(extraction_type), # The kind of extraction method used
           extraction_type_class = as.factor(extraction_type_class), # The kind of extraction method used, fewer levels than extraction_type
           region = as.factor(region), # 21 regions: islands and some small regions (fewer residents) are not included
           installer = as.factor(installer), # Organization that installed the well
           )

# Load the test data
TestData <- readr::read_csv("TestData.csv")
# Filter and format the test data
DataTest_2_1 <- TestData %>% 
    select(amount_tsh, gps_height, longitude, latitude, region, population, construction_year, extraction_type,
           management_group, payment, quality_group, quantity, source_type, source_class, waterpoint_type, installer, extraction_type_class) %>%
    mutate(waterpoint_type = as.factor(waterpoint_type), # amount of water available
           source_type = as.factor(source_type), # The source of the water, since the variable wasnt that important in the first model, 
                                       # we use source_type because it has less levels
           source_class = as.factor(source_class), # were the water is "stored", groundwater or surface, maybe drop this
           quantity = as.factor(quantity), # quantity of water available / quantity = quantity_group
           quality_group = as.factor(quality_group), # quality of the water, fewer levels than water_quality, in the first model we used water_quality
                                                     # but the variable was not that important                     
           payment = as.factor(payment), # What the water costs / payment_type = payment the level names are different but the variable is the same
           management_group = as.factor(management_group), # How the waterpoint is managed / management_group has less levels than management, not that important in first model
           extraction_type = as.factor(extraction_type), # The kind of extraction method used
           extraction_type_class = as.factor(extraction_type_class), # The kind of extraction method used, fewer levels than extraction_type
           region = as.factor(region), # 21 regions: islands and some small regions (fewer residents) are not included
           installer = as.factor(installer), # Organization that installed the well
           )



Data_training_hyper <- Data_factor_2
Data_training_hyper$installer <- as.character(Data_training_hyper$installer) # Convert installer to character
Data_training_hyper$installer[is.na(Data_training_hyper$installer)] <- "unknown" # Replace NA values in installer with "unknown"
Data_training_hyper$installer <- as.factor(Data_training_hyper$installer) # Ensure installer is a factor
Data_training_hyper$status_group <- factor(Data_training_hyper$status_group,
    levels = c("functional", "functional needs repair", "non functional"),
    labels = c("functional", "functional_needs_repair", "non_functional"))



tune_grid <- expand.grid(
  .mtry = c(1, 2, 4, 6, 8, 10), # Values for mtry
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
  data = Data_training_hyper,
  method = "ranger",
  tuneGrid = tune_grid,
  trControl = train_control,
  metric = "Kappa", # Use Kappa for multi-class classification
  importance = "impurity", # To get feature importance later
  num.trees = 500, # Fixed number of trees for this tuning
  respect.unordered.factors = TRUE, # Respect unordered factors
  seed = 24 # For reproducibility of random processes in ranger
)

Data_training_hyper %>%
  summarise(across(everything(), ~ sum(is.na(.))))
# View the results
print(tuned_model)

# Plot the tuning results (e.g., performance across mtry values)
png("ParameterGridSearch.png",    # The name of the file
    width = 1200,                               # Width in pixels
    height = 800,                              # Height in pixels
    res = 100)                                  # Resolution (dots per inch)
plot(tuned_model)
dev.off()


best_params <- tuned_model$bestTune
best_params

# Get the final model (trained on the full dataset with best parameters)
final_model <- tuned_model$finalModel
print(final_model)

# Replace Na value in installer with "unknown"
DataTest_2_1$installer <- as.character(DataTest_2_1$installer)
DataTest_2_1$installer[is.na(DataTest_2_1$installer)] <- "unknown" # Replace NA values in installer with "unknown"
DataTest_2_1$installer <- as.factor(DataTest_2_1$installer) # Ensure installer is a factor

# Predict on new data
pred <- predict(final_model, data = DataTest_2_1, type = "response")
# Save the predictions to a CSV file
submission <- data.frame(id = TestData$id, status_group = pred$predictions)
write.csv(submission, file = "submissionRandomForestGridSearch.csv", row.names = FALSE)



tune_grid_2 <- expand.grid(
  .mtry = c(4, 8, 12, 16), # Values for mtry
  .splitrule = c("gini"), # Values for splitrule
  .min.node.size = c(1) # Values for min.node.size
)

tuned_model_2 <- train(
  status_group ~ .,
  data = Data_training_hyper,
  method = "ranger",
  tuneGrid = tune_grid_2,
  trControl = train_control,
  metric = "Kappa", # Use Kappa for multi-class classification
  importance = "impurity", # To get feature importance later
  num.trees = 500, # Fixed number of trees for this tuning
  respect.unordered.factors = TRUE, # Respect unordered factors
  seed = 24 # For reproducibility of random processes in ranger
)
print(tuned_model_2)
plot(tuned_model_2)


# Train model
model_upVar_best <- ranger(
  status_group ~ ., 
  data = Data_training_hyper,
  num.trees = 5000,
  mtry = 16,
  min.node.size = 1,
  replace = TRUE,
  splitrule = "gini",
  respect.unordered.factors = TRUE,
  seed = 24, importance = "permutation"
)
model_upVar_best$prediction.error

DataTest_2_1$installer <- as.character(DataTest_2_1$installer) # Convert installer to character
DataTest_2_1$installer[is.na(DataTest_2_1$installer)] <- "unknown" # Replace NA values in installer with "unknown"
DataTest_2_1$installer <- as.factor(DataTest_2_1$installer) # Ensure installer is a factor


pred <- predict(model_upVar_best, data = DataTest_2_1, type = "response")

# Save the predictions to a CSV file
submission <- data.frame(id = TestData$id, status_group = pred$predictions)
submission$status_group <- gsub("_", " ", submission$status_group)
write.csv(submission, file = "submissionRandomForestGridSearchBestParameters.csv", row.names = FALSE)


Data_training_hyper_min <- Data_training_hyper %>%
  select(-quality_group, source_class, management_group)

# Train model with fewer variables
model_upVar_min <- ranger(
  status_group ~ ., 
  data = Data_training_hyper_min,
  num.trees = 5000,
  mtry = 16,
  min.node.size = 1,
  replace = TRUE,
  splitrule = "gini",
  respect.unordered.factors = TRUE,
  seed = 24, importance = "permutation"
)
model_upVar_min$prediction.error


TestData_min <- TestData_2_1 %>%
  select(-quality_group, source_class, management_group)
# Predict on the test data with fewer variables
pred <- predict(model_upVar_min, data = DataTest_2_1, type = "response")

# Save the predictions to a CSV file
submission <- data.frame(id = TestData$id, status_group = pred$predictions)
write.csv(submission, file = "submissionRandomForestGridSearchBestParametersMIN.csv", row.names = FALSE)
