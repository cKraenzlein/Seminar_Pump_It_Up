# Seminar
# Pump it Up
This repository contains the data and models for the DrivenData competition Pump It Up.
# Getting Started
How to run the code:
1. Open the Tuning File for the desired Learner (XGBoost, RandomForest, LightGBM, CATBoost)
2. Run this file, it will create 3 Plots that will be saved in the Plot folder, one representing the tuning batches and the other two the feature importance of the tuned model, the tuned model will be saved into the FinalModels folder
3. In the next step you need to run the Priticting file for the same learner, it will create an csv-file containing the predicted data

   
# Structur:
## Data
* contains the 3 csv-files, provided by the competition site
* two R files for the preprocessing steps
## HyperparameterTuning:
* Folder Results:
  * contains the tuning results of the final models
  * if the rds-file from the complete tuning was to large to push to git, only the hyperparameter of th models are available
* R file for each model containing the imputation and encoding pipeline, as well as the tuning implementation
## FinalModels
* contains the final model as rds-file
## Predicting
* contains R-files for predicting the test data
## Submissions
* contains the csv-files with the predicted data used for model evaluation
## Plots
* contains Plots used for the report
## Additional
* contains files used for creating plots or analyzing the data


## Requirements:
* mlr3
* mlr3pipeline
* mlr3tuning
* mlr3learners
* mlr3extralearner
* mlr3viz
* mlr3mbo
* ranger
* catboost
* xgBoost
* lightgbm
* VIM
* dplyr
* ggplot2
* here
* 

