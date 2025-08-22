## xgBoost Model for Classification
# Load required libraries
library(xgboost)
library(Matrix)
library(MatrixModels)
library(data.table)
library(dplyr)

#source the training data
here::here()
source("CreateTrainingData.R")
source("CreateTestDataSet.R")
# Load the training data
Data <- Trainingsdata_xgBoost
label <- as.numeric(Data$status_group)
Data_test <- Testdata_xgBoost

# Split data into training and testing sets
train_index <- caret::createDataPartition(Data$status_group, p = 0.7, list = FALSE)
train_data <- Data[train_index, ]
test_data <- Data[-train_index, ]

# Remove the target variable from the training data
Data <- Data %>% select(-status_group)

# Convert data frames to numeric matrices. Xgboost requires user to enter data as a numeric matrix
data_train_matrix <- as.matrix(as.data.frame(lapply(Data, as.numeric)))
data_test_matrix <- as.matrix(as.data.frame(lapply(Data_test, as.numeric)))

# Create a xgb.DMatrix which is the best format to use to create an xgboost model
train.DMatrix <- xgb.DMatrix(data = data_train_matrix, label = label, missing = NA)

# Run the Model x times to improve model performance

# create data frame to store results
solution.table <- data.frame(matrix(NA, nrow = length(Id), ncol = 11))
solution.table[ , 1] <- Id

for (i in 2:11) {
    # Set seed so that the results are reproducible
    set.seed(i)

    #Create model using the same parameters used in xgb.cv
    model <- xgb.train(data = train.DMatrix, objective = "multi:softmax", booster = "gbtree",
                    eval_metric = "merror", nrounds = 500, 
                    num_class = 4,eta = .2, max_depth = 12, colsample_bytree = .4)

    #Predict. Used the data_test.noID because it contained the same number of columns as the train.DMatrix
    #used to build the model.
    predict <- predict(model, data_test_matrix)

    #Modify prediction labels to match submission format
    predict[predict==1]<-"functional"
    predict[predict==2]<-"functional needs repair"
    predict[predict==3]<-"non functional"

    #View prediction
    table(predict)

    #Add the solution to column i of the solutions data frame. This creates a data frame with a column for
    #each prediction set. Each prediction is a vote for that prediction. Next I will count the number of votes
    #for each prediction as use the element with the most votes as my final solution.
    solution.table[ , i] <- predict
}

#Count the number of votes for each solution for each row
solution.table.count <- apply(solution.table, MARGIN = 1, table)

#Create a vector to hold the final solution
predict.combined <- vector()

x <- 1
#Finds the element that has the most votes for each prediction row
for (x in 1 : nrow(Data_test)){
  predict.combined[x] <- names(which.max(solution.table.count[[x]]))}

#View the number of predictions for each classification
table(predict.combined)

#Create solution data frame
solution <- data.frame(id = Id, status_group = predict.combined)

#View the first five rows of the solution to ensure that it follows submission format rules
head(solution)

#Create csv submission file
write.csv(solution, file = "xgBoost_1.csv", row.names = FALSE)

#Calculate the importance of each variable to the model.
#Used this function to remove variables from the model variables which don't contribute to the model.
importance <- xgb.importance(feature_names = colnames(Data), model = model)
importance
xgb.plot.importance(importance_matrix = importance)


# xgBoost with hyperparameter tuning

# Split data into hypertraining and training sets
train_index <- caret::createDataPartition(Data$status_group, p = 0.7, list = FALSE)
Train_data <- Data[train_index, ]
Hypertrain_data <- Data[-train_index, ]
label_train <- as.numeric(Train_data$status_group)
label_hypertrain <- as.numeric(Hypertrain_data$status_group)

# Remove the target variable from the training data
Train_data <- Train_data %>% select(-status_group)
Hypertrain_data <- Hypertrain_data %>% select(-status_group)

# Convert data frames to numeric matrices. Xgboost requires user to enter data as a numeric matrix
Train_data_matrix <- as.matrix(as.data.frame(lapply(Train_data, as.numeric)))
Hypertrain_data_matrix <- as.matrix(as.data.frame(lapply(Hypertrain_data, as.numeric)))

# Create a xgb.DMatrix which is the best format to use to create an xgboost model
train.DMatrix <- xgb.DMatrix(data = Train_data_matrix, label = label_train, missing = NA)
hypertrain.DMatrix <- xgb.DMatrix(data = Hypertrain_data_matrix, label = label_hypertrain, missing = NA)


num_classes <- length(unique(label_train)) # Get the number of classes in the target variable
#default parameters
params <- list( booster = "gbtree", objective =  "multi:softmax", eta = 0.3, base_score = 0.5, num_class = num_classes, 
                gamma = 0, max_depth = 6, min_child_weight = 1, subsample = 1, colsample_bytree = 1)

xgbcv <- xgb.cv(    params = params, data = train.DMatrix, nrounds = 100, nfold = 5, showsd = T,
                    stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)


# xgBioost with mlr3
# Load required libraries
library(mlr3verse)
library(mlr3proba)
library(xgboost)
library(tidyverse)
library(survival)

set.seed(1)

Data_mlr3 <- Trainingsdata_xgBoost

# load learner and set search space
learner = lts(lrn("classif.xgboost"))

# retrieve task
task <- as_task_classif(Data_mlr3, target = "status_group")

# load tuner and set batch size
tuner <- tnr("random_search", batch_size = 10)

# hyperparameter tuning on the pima data set
instance = ti(
  tuner = tnr("grid_search", resolution = 5, batch_size = 25),
  task = task,
  learner = learner,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
)


tuner$optimize(instance)

# best performing hyperparameter configuration
instance$result

# ------------------------------------------------------------------------------------------------------------------------------------ #
# Load necessary libraries
library(proxy)
library(mlr3)
library(rpart.plot)
library(mlr3learners)
library(data.table)
library(mlr3verse)
library(mlr3filters)
library(mlr3viz)
library(mlr3tuning)
library(ggplot2)


# Set seed for reproducibility
set.seed(24)

# Load training data
Data_mlr3 <- Trainingsdata_xgBoost %>%
  mutate(across(where(is.character), as.factor)) %>% 
  mutate(year_recorded = as.numeric(format(date_recorded, "%Y"))) %>% 
  select(-date_recorded)

str(Data_mlr3)

# create learning task
task_wells <- as_task_classif(status_group ~ ., data = Data_mlr3)
task_wells

# load learner and set hyperparameter
learner = lrn("classif.rpart", keep_model = TRUE, maxdepth = 50, minsplit = 10, cp = 0.01)

# Train the model
learner$train(task_wells)

# Plot the decision tree
rpart.plot(learner$model)

autoplot(
  learner,
  type = "ggparty"
)




# Filter for feature importance
learner <- lrn("classif.ranger", importance = "permutation", "oob.error" = TRUE)
filter <- flt("importance", learner = learner)
filter$calculate(task_wells)
head(as.data.table(filter), 5)


