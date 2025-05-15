# Load necessary packages
library(readr)
library(dplyr)
library(here)
# Set the working directory to the location of the script
here::here()
# Read in the Datasets
TrainingData_Variables <- read_csv("TrainingData_Variables.csv")
TrainingData_TargetVariable <- read_csv("TrainingData_TargetVariable.csv")
# Combine the two datasets by the column "id"
TrainingData <- merge(x = TrainingData_Variables, y = TrainingData_TargetVariable, by = "id")
