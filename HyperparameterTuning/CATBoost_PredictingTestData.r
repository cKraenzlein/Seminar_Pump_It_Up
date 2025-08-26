# Set the working directory
here::here()
# Source the test data
source("./Data/CreateTestDataSet.R")

# Set seed for reproducibility
set.seed(24)

# Save the ID column and the test data
ID <- Id
Test_Data <- Data_Test_Final

# Load the final model
final_learner <- readRDS("./FinalModels/catboost_final_model.rds")

# Predict on the test set
prediction = final_learner$predict_newdata(Test_Data)

table(prediction$response)

pred <- prediction$response
# Save the predictions to a CSV file
submission <- data.frame(id = ID, status_group = pred)
write.csv(submission, file = "CatBoost_Tuned.csv", row.names = FALSE)
