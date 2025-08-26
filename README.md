# Seminar
# Pump it Up
This repository contains the data and the code to reproduce the models for the DrivenData competition Pump It Up.
# Getting Started
How to run the code:
1. Open the Tuning File for the desired Learner (XGBoost, RandomForest, LightGBM, CATBoost)
2. Run this file, it will create 1 Plots that will be saved in the Plot folder, representing the feature importance of the tuned model, the tuned model will be saved into the FinalModels folder.
3. In the next step you need to run the Preticting file for the same learner, it will create an csv-file containing the predicted data.

   
# Structur:
## Data
* contains the 3 csv-files, provided by the competition site
* two R files for the preprocessing steps
## HyperparameterTuning:
* Folder Results:
  * should contain the tuning results of the final models after training them
* R file for each model containing the imputation and encoding pipeline, as well as the tuning implementation
## FinalModels
* should contain final model after running hyperparameter tuning
## Predicting
* contains R-files for predicting the test data
## Submissions
* contains the csv-files with the predicted data used for model evaluation
## Plots
* contains Plots used for the report
## Remaining
* contains files used for creating plots or analyzing the data
## Loose in the repo
R File for the creation of the plots in the report
R File containing the Imputation pipeline

The RDS files from our tuning and our final models were too large for pushing to git.
Since we unified the tuning algorithm in the latest stages, running the algorithm could result in different results, because we adjusted small things like the Hyperparameterspace and resampling strategy. We did not tune all models again, because we wanted to focus on our report.

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

