# Load packages
library(dplyr)
library(ranger)   # a c++ implementation of random forest 

# Source the Training Data
here::here()
source("CreateTrainingData.R")
# Load the training data
Data <- TrainingData

# Filter the data
Data_factor <- Data %>% 
    select(amount_tsh, funder, gps_height, longitude, latitude, region, population, permit, construction_year, extraction_type,
           management_group, payment, quality_group, quantity, source_type, source_class, waterpoint_type, status_group, installer, extraction_type_class, scheme_management) %>%
    mutate(status_group = as.factor(status_group), 
           waterpoint_type = as.factor(waterpoint_type), # amount of water available
           source_type = as.factor(source_type), # The source of the water, since the variable wasnt that important in the first model, 
                                       # we use source_type because it has less levels
           source_class = as.factor(source_class), # were the water is "stored", groundwater or surface, maybe drop this
           scheme_management = as.factor(scheme_management), # Who operates the waterpoint, fewer levels than scheme_name
           quantity = as.factor(quantity), # quantity of water available / quantity = quantity_group
           quality_group = as.factor(quality_group), # quality of the water, fewer levels than water_quality, in the first model we used water_quality
                                                     # but the variable was not that important                     
           payment = as.factor(payment), # What the water costs / payment_type = payment the level names are different but the variable is the same
           management_group = as.factor(management_group), # How the waterpoint is managed / management_group has less levels than management, not that important in first model
           extraction_type = as.factor(extraction_type), # The kind of extraction method used
           extraction_type_class = as.factor(extraction_type_class), # The kind of extraction method used, fewer levels than extraction_type
           region = as.factor(region), # 21 regions: islands and some small regions (fewer residents) are not included
           installer = as.factor(installer), # Organization that installed the well
           funder = as.factor(funder)) # Who funded the well

n_features <- length(setdiff(names(Data_factor), "status_group"))

# modell
mes_rf1 <- ranger(
  status_group ~ ., 
  data = Data_factor,
  mtry = floor(n_features / 3),
  respect.unordered.factors = "order",
  seed = 123
)

default_rmse <- sqrt(mes_rf1$prediction.error)

# Tuning
# create hyperparameter grid
hyper_grid <- expand.grid(
  mtry = floor(n_features * c(.05, .15, .25, .333, .4)),
  min.node.size = c(1, 3, 5, 10), 
  replace = c(TRUE, FALSE),                               
  sample.fraction = c(.5, .63, .8),                       
  rmse = NA                                               
)

# execute full cartesian grid search
for(i in seq_len(nrow(hyper_grid))) {
  # fit model for ith hyperparameter combination
  fit <- ranger(
    formula         = status_group ~ ., 
    data            = Data_factor, 
    num.trees       = n_features * 10,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$min.node.size[i],
    replace         = hyper_grid$replace[i],
    sample.fraction = hyper_grid$sample.fraction[i],
    verbose         = FALSE,
    seed            = 123,
    respect.unordered.factors = 'order',
  )
  # export OOB error 
  hyper_grid$rmse[i] <- sqrt(fit$prediction.error)
}

# assess top 10 models
top_models <- hyper_grid %>%
  arrange(rmse) %>%
  mutate(perc_gain = (default_rmse - rmse) / default_rmse * 100) %>%
  head(10)
top_models


# best modell -------------------------------------------------------------------------------------------------------------------------
mes_rf1 <- ranger(
  status_group ~ ., 
  data = Data_factor,
  num.trees = 10000,
  mtry = top_models$mtry[1],
  min.node.size = top_models$min.node.size[1],
  replace = top_models$replace[1],
  sample.fraction = top_models$sample.fraction[1],
  respect.unordered.factors = "order",
  seed = 123, importance = "permutation"
)

# get OOB RMSE
(default_rmse <- sqrt(mes_rf1$prediction.error))
importance(mes_rf1)

# Variable importance
importance_scores <- mes_rf1$variable.importance
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

ggsave(importance_Variables, file = "importance_Variables_Updated.png", width = 14, height = 8)

# Ist Modell nutzlos?
set.seed(123)
random_guesses <- sample(levels(Data_factor$status_group), size = length(Data_factor$status_group), replace = TRUE, prob = prop.table(table(Data_factor$status_group)))

# Calculate accuracy of random guessing
random_guess_accuracy <- mean(random_guesses == Data_factor$status_group)
print(random_guess_accuracy)

# Extract the model's OOB accuracy
ranger_accuracy <- mes_rf1$prediction.error
model_accuracy <- 1 - ranger_accuracy
print(model_accuracy)

# Compare the two
cat("Baseline Accuracy:", random_guess_accuracy, "\n")
cat("Random Forest Accuracy:", model_accuracy, "\n")


sort(colSums(is.na(Data_factor)))

# Predict the test data
# Load the test data
TestData <- readr::read_csv("TestData.csv")

TestData_2 <- TestData %>% 
    select(amount_tsh, funder, gps_height, longitude, latitude, region, population, permit, construction_year, extraction_type,
           management_group, payment, quality_group, quantity, source_type, source_class, waterpoint_type, installer, extraction_type_class, scheme_management) %>%
    mutate(waterpoint_type = as.factor(waterpoint_type), # amount of water available
           source_type = as.factor(source_type), # The source of the water, since the variable wasnt that important in the first model, 
                                       # we use source_type because it has less levels
           source_class = as.factor(source_class), # were the water is "stored", groundwater or surface, maybe drop this
           scheme_management = as.factor(scheme_management), # Who operates the waterpoint, fewer levels than scheme_name
           quantity = as.factor(quantity), # quantity of water available / quantity = quantity_group
           quality_group = as.factor(quality_group), # quality of the water, fewer levels than water_quality, in the first model we used water_quality
                                                     # but the variable was not that important                     
           payment = as.factor(payment), # What the water costs / payment_type = payment the level names are different but the variable is the same
           management_group = as.factor(management_group), # How the waterpoint is managed / management_group has less levels than management, not that important in first model
           extraction_type = as.factor(extraction_type), # The kind of extraction method used
           extraction_type_class = as.factor(extraction_type_class), # The kind of extraction method used, fewer levels than extraction_type
           region = as.factor(region), # 21 regions: islands and some small regions (fewer residents) are not included
           installer = as.factor(installer), # Organization that installed the well
           funder = as.factor(funder)) # Who funded the well


pred <- predict(mes_rf1, data = TestData_2, type = "response")

# Save the predictions to a CSV file
submission <- data.frame(id = TestData$id, status_group = pred$predictions)
write.csv(submission, file = "submissionRandomForestUpdatedVariableSet.csv", row.names = FALSE)



# Multokillinearitaet anschauen mithilfe von Korrelationsmatrix
# Load the necessary package
library(corrplot)

# Calculate the correlation matrix
X <- Data_factor %>% 
  select(-status_group) %>% 
  mutate_if(is.character, as.factor) %>% 
  mutate_if(is.factor, as.numeric)

correlation_matrix <- cor(X, use = "pairwise.complete.obs")
# Visualize the correlation matrix  
png("correlation_plot_updatedVariables.png",    # The name of the file
    width = 1200,                               # Width in pixels
    height = 1200,                              # Height in pixels
    res = 100)                                  # Resolution (dots per inch)
corrplot(correlation_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45, addCoef.col = "black", number.cex = 0.7)
dev.off()


# Remove Management_group, extraction_type_class, and funder

Data_factor_2 <- Data_factor %>% 
  select(-management_group, -extraction_type_class, -funder, -scheme_management, -permit)

DataTest_2_1 <- TestData_2 %>% 
  select(-management_group, -extraction_type_class, -funder, -scheme_management, -permit)


max_trees <- 10000
# Refit the model with the reduced set of variables
UpdatedVar <- ranger(
  status_group ~ ., 
  data = Data_factor,
  num.trees = max_trees,
  mtry = top_models$mtry[1],
  min.node.size = top_models$min.node.size[1],
  replace = top_models$replace[1],
  sample.fraction = top_models$sample.fraction[1],
  respect.unordered.factors = "order",
  oob.error = TRUE,
  seed = 123, importance = "permutation"
)
# get OOB RMSE
(default_rmse <- sqrt(UpdatedVar$prediction.error))
importance(UpdatedVar)

# Variable importance
importance_scores <- UpdatedVar$variable.importance
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


# --------------------------------------------------------------


n_features_upVar <- length(setdiff(names(Data_factor_2), "status_group"))
# modell
model_upVar <- ranger(
  status_group ~ ., 
  data = Data_factor_2,
  mtry = floor(sqrt(n_features_upVar)),
  respect.unordered.factors = "order",
  seed = 123
)

default_rmse_UpVar <- sqrt(model_upVar$prediction.error)

# Tuning
# create hyperparameter grid
hyper_grid <- expand.grid(
  mtry = floor(n_features * c(.05, .15, .25, .333, .4)),
  min.node.size = c(1, 3, 5, 10), 
  replace = c(TRUE, FALSE),                               
  sample.fraction = c(.5, .63, .8),                       
  rmse = NA                                               
)

# execute full cartesian grid search
for(i in seq_len(nrow(hyper_grid))) {
  # fit model for ith hyperparameter combination
  fit <- ranger(
    formula         = status_group ~ ., 
    data            = Data_factor_2, 
    num.trees       = 1000,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$min.node.size[i],
    replace         = hyper_grid$replace[i],
    sample.fraction = hyper_grid$sample.fraction[i],
    verbose         = FALSE,
    seed            = 123,
    respect.unordered.factors = 'order',
  )
  # export OOB error 
  hyper_grid$rmse[i] <- sqrt(fit$prediction.error)
}

# assess top 10 models
top_models_upVar <- hyper_grid %>%
  arrange(rmse) %>%
  mutate(perc_gain = (default_rmse - rmse) / default_rmse_UpVar * 100) %>%
  head(10)
top_models_upVar


# best modell -------------------------------------------------------------------------------------------------------------------------
model_upVar_best <- ranger(
  status_group ~ ., 
  data = Data_factor_2,
  num.trees = 10000,
  mtry = top_models_upVar$mtry[1],
  min.node.size = top_models_upVar$min.node.size[1],
  replace = top_models_upVar$replace[1],
  sample.fraction = top_models_upVar$sample.fraction[1],
  respect.unordered.factors = "order",
  seed = 123, importance = "permutation"
)

model_upVar_best$prediction.error

pred <- predict(model_upVar_best, data = DataTest_2_1, type = "response")

# Save the predictions to a CSV file
submission <- data.frame(id = TestData$id, status_group = pred$predictions)
write.csv(submission, file = "submissionRandomForestUpdatedVariableSet2.csv", row.names = FALSE)

# -----------------------------------------------------------------------------------------------------------
# Grid search for the best hyperparameters with caret
library(caret)
set.seed(24)
Data_traing_hyper$installer[is.na(Data_traing_hyper$installer)] <- "unknown" # Replace NA values in installer with "unknown"
Data_traing_hyper$installer <- as.factor(Data_traing_hyper$installer) # Ensure installer is a factor
Data_traing_hyper <- na.omit(Data_factor_2)
Data_traing_hyper$status_group <- factor(Data_traing_hyper$status_group,
    levels = c("functional", "functional needs repair", "non functional"),
    labels = c("functional", "functional_needs_repair", "non_functional"))



tune_grid <- expand.grid(
  .mtry = c(2, 4, 6),                 # Values for mtry
  .splitrule = c("gini", "extratrees"), # Values for splitrule
  .min.node.size = c(1, 3)             # Values for min.node.size
)

train_control <- trainControl(
  method = "cv",                # Cross-validation
  number = 5,                   # 5 folds
  verboseIter = TRUE,           # Show progress
  classProbs = TRUE,            # Needed for ROC curve if desired
  summaryFunction = multiClassSummary # Use multiClassSummary for > 2 classes
)

tuned_model <- train(
  status_group ~ .,
  data = Data_traing_hyper,
  method = "ranger",
  tuneGrid = tune_grid,
  trControl = train_control,
  metric = "ROC", # For binary classification: "Accuracy", "Kappa", "ROC" etc.
  importance = "impurity", # To get feature importance later
  num.trees = 500, # Fixed number of trees for this tuning
  seed = 24 # For reproducibility of random processes in ranger
)

# View the results
print(tuned_model)

# Plot the tuning results (e.g., performance across mtry values)
plot(tuned_model)

best_params <- tuned_model$bestTune
best_params

# Get the final model (trained on the full dataset with best parameters)
final_model <- tuned_model$finalModel
print(final_model)

# For categorical features: use mode (most frequent category)
get_mode <- function(v) {
  uniqv <- unique(v[!is.na(v)])
  if (length(uniqv) == 0) return(NA) # Handle case where all are NA
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Replace Na value in installer with "unknown"
DataTest_2_1$installer[is.na(DataTest_2_1$installer)] <- "unknown" # Replace NA values in installer with "unknown"
DataTest_2_1$$installer <- as.factor(DataTest_2_1$installer) # Ensure installer is a factor

# Predict on new data
pred <- predict(final_model, data = DataTest_2_1, type = "response")
# Save the predictions to a CSV file
submission <- data.frame(id = TestData$id, status_group = pred$predictions)
write.csv(submission, file = "submissionRandomForestGridSearch.csv", row.names = FALSE)
