# Random Forest Model with fewer variables
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
Data_train <- Data %>% 
    select( status_group, amount_tsh, gps_height, longitude, latitude, region, population, construction_year, extraction_type,
           management_group, payment, quality_group, quantity, source_type, source_class, waterpoint_type, installer, extraction_type_class) %>%
    mutate(status_group = as.factor(status_group), 
           waterpoint_type = as.factor(waterpoint_type), # amount of water available
           source_type = as.factor(source_type), # The source of the water, since the variable wasnt that important in the first model, 
                                       # we use source_type because it has less levels
           source_class = as.factor(source_class), # were the water is "stored", groundwater or surface, maybe drop this
           quantity = factor(quantity, ordered = TRUE, levels = c("enough", "seasonal", "insufficient", "dry", "unknown")), 
           # quantity of water available / quantity = quantity_group
           quality_group = factor(quality_group, ordered = TRUE, levels = c("good", "milky", "colored", "fluoride", "salty", "unknown")), 
           # quality of the water, fewer levels than water_quality, in the first model we used water_quality
                                                     # but the variable was not that important                     
           payment = factor(payment, ordered = TRUE, levels = c("pay per bucket", "pay monthly", "pay annually", "pay when scheme fails", "other", "unknown", "never pay")), 
           # What the water costs / payment_type = payment the level names are different but the variable is the same
           management_group = as.factor(management_group), # How the waterpoint is managed / management_group has less levels than management, not that important in first model
           extraction_type = as.factor(extraction_type), # The kind of extraction method used
           extraction_type_class = as.factor(extraction_type_class), # The kind of extraction method used, fewer levels than extraction_type
           region = as.factor(region), # 21 regions: islands and some small regions (fewer residents) are not included
           installer = as.factor(installer), # Organization that installed the well
           )

Data_train$installer <- as.character(Data_train$installer)
Data_train$installer[is.na(Data_train$installer)] <- "unknown" # Replace NA values in installer with "unknown"
Data_train$installer <- as.factor(Data_train$installer) # Ensure installer is a factor

# Load the test data
TestData <- readr::read_csv("TestData.csv")
# Filter and format the test data
Data_test <- TestData %>% 
    select(amount_tsh, gps_height, longitude, latitude, region, population, construction_year, extraction_type,
           management_group, payment, quality_group, quantity, source_type, source_class, waterpoint_type, installer, extraction_type_class) %>%
    mutate(waterpoint_type = as.factor(waterpoint_type), # amount of water available
           source_type = as.factor(source_type), # The source of the water, since the variable wasnt that important in the first model, 
                                       # we use source_type because it has less levels
           source_class = as.factor(source_class), # were the water is "stored", groundwater or surface, maybe drop this
           quantity = factor(quantity, ordered = TRUE, levels = c("enough", "seasonal", "insufficient", "dry", "unknown")), 
           # quantity of water available / quantity = quantity_group
           quality_group = factor(quality_group, ordered = TRUE, levels = c("good", "milky", "colored", "fluoride", "salty", "unknown")), 
           # quality of the water, fewer levels than water_quality, in the first model we used water_quality
                                                     # but the variable was not that important                     
           payment = factor(payment, ordered = TRUE, levels = c("pay per bucket", "pay monthly", "pay annually", "pay when scheme fails", "other", "unknown", "never pay")), 
           # What the water costs / payment_type = payment the level names are different but the variable is the same
           management_group = as.factor(management_group), # How the waterpoint is managed / management_group has less levels than management, not that important in first model
           extraction_type = as.factor(extraction_type), # The kind of extraction method used
           extraction_type_class = as.factor(extraction_type_class), # The kind of extraction method used, fewer levels than extraction_type
           region = as.factor(region), # 21 regions: islands and some small regions (fewer residents) are not included
           installer = as.factor(installer), # Organization that installed the well
           )

# --------------------------------------------------------------
n_features <- length(setdiff(names(Data_test), "status_group"))
# modell
model_upVar <- ranger(
  status_group ~ ., 
  data = Data_train,
  mtry = floor(sqrt(n_features)),
  respect.unordered.factors = TRUE,
  seed = 24
)

default_rmse <- sqrt(model_upVar$prediction.error)

# Tuning
# create hyperparameter grid
hyper_grid <- expand.grid(
  mtry = floor(n_features * c(.05, .15, .25, .333, .4, .5)),
  min.node.size = c(1, 3, 5, 10, 20),
  splitrule = c("gini", "extratrees"), 
  replace = TRUE,                               
  sample.fraction = c(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
  respect.unordered.factors = TRUE,                     
  rmse = NA                                               
)

# execute full cartesian grid search
for(i in seq_len(nrow(hyper_grid))) {
  # fit model for ith hyperparameter combination
  fit <- ranger(
    formula         = status_group ~ ., 
    data            = Data_train, 
    num.trees       = 1000,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$min.node.size[i],
    splitrule       = hyper_grid$splitrule[i],
    replace         = TRUE,
    sample.fraction = hyper_grid$sample.fraction[i],
    verbose         = FALSE,
    seed            = 24,
    respect.unordered.factors = TRUE,
  )
  # export OOB error 
  hyper_grid$rmse[i] <- sqrt(fit$prediction.error)
}

# assess top 10 models
top_models_upVar <- hyper_grid %>%
  arrange(rmse) %>%
  mutate(perc_gain = (default_rmse - rmse) / default_rmse * 100) %>%
  head(10)
top_models_upVar


# best modell -------------------------------------------------------------------------------------------------------------------------
model_upVar_best <- ranger(
  status_group ~ ., 
  data = Data_train,
  num.trees = 5000,
  mtry = top_models_upVar$mtry[1],
  min.node.size = top_models_upVar$min.node.size[1],
  replace = top_models_upVar$replace[1],
  sample.fraction = top_models_upVar$sample.fraction[1],
  respect.unordered.factors = TRUE,
  seed = 24, importance = "permutation"
)

model_upVar_best$prediction.error

pred <- predict(model_upVar_best, data = Data_test, type = "response")

# Save the predictions to a CSV file
submission <- data.frame(id = TestData$id, status_group = pred$predictions)
write.csv(submission, file = "submissionRandomForestGridSearch.csv", row.names = FALSE)

# Use the base model without grid search
pred <- predict(model_upVar, data = Data_test, type = "response")

# Save the predictions to a CSV file
submission <- data.frame(id = TestData$id, status_group = pred$predictions)
write.csv(submission, file = "submissionRandomForestWithoutGridSearch.csv", row.names = FALSE)







# Replace Na value in installer with "unknown"
Data_train$installer <- as.character(Data_train$installer)
Data_train$installer[is.na(Data_train$installer)] <- "unknown" # Replace NA values in installer with "unknown"
Data_train$installer <- as.factor(Data_train$installer) # Ensure installer is a factor
# Predict on new data
model_upVar <- ranger(
  status_group ~ ., 
  data = Data_train,
  mtry = floor(sqrt(n_features)),
  respect.unordered.factors = TRUE,
  min.node.size = 1,
  seed = 24
)
model_upVar$prediction.error


