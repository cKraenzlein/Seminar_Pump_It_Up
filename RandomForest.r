# Load packages
library(dplyr)
library(ranger)   # a c++ implementation of random forest 
#library(h2o)      # a java-based implementation of random forest

# Source the Training Data
here::here()
source("CreateTrainingData.R")
# Load the training data
Data <- TrainingData

str(Data)

# Filter the data
Data_factor <- Data %>% 
    select(amount_tsh, funder, gps_height, longitude, latitude, region, population, permit, construction_year, extraction_type,
           management, payment, water_quality, quantity, source, waterpoint_type, status_group) %>%
    mutate(status_group = as.factor(status_group), 
           waterpoint_type = as.factor(waterpoint_type),
           source = as.factor(source),
           quantity = as.factor(quantity),
           water_quality = as.factor(water_quality), 
           payment = as.factor(payment), 
           management = as.factor(management), 
           extraction_type = as.factor(extraction_type),
           region = as.factor(region), 
           funder = as.factor(funder))

# Check the structure of the data
str(Data_factor)

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
  num.trees = n_features*10,
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
ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance", x = "Variable", y = "Importance")


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
           management, payment, water_quality, quantity, source, waterpoint_type) %>%
    mutate(waterpoint_type = as.factor(waterpoint_type),
           source = as.factor(source),
           quantity = as.factor(quantity),
           water_quality = as.factor(water_quality), 
           payment = as.factor(payment), 
           management = as.factor(management), 
           extraction_type = as.factor(extraction_type),
           region = as.factor(region), 
           funder = as.factor(funder))

pred <- predict(mes_rf1, data = TestData_2, type = "response")

# Save the predictions to a CSV file
submission <- data.frame(id = TestData$id, status_group = pred$predictions)
write.csv(submission, file = "submissionRandomForest.csv", row.names = FALSE)
