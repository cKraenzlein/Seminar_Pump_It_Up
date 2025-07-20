# Load necessary packages
library(readr)
library(dplyr)
# Set the working directory to the location of the script
here::here()
# Load the test data
TestData <- readr::read_csv("TestData.csv")
# Source Trainings Data for mean_construction_year and Top25_installers and mean_logitude and mean_latitude
source("CreateTrainingData.r")
# Save Id column for later use
Id <- TestData$id
# Fromat the test data
 TestData <-  TestData %>%
    left_join(Mean_Location, by = c("region", "district_code"))

TestData_Formatted <- TestData %>%
    select(-payment, -region_code, -quantity_group, -scheme_management, -extraction_type, -water_quality) %>% # Take the Training data and remove all duplicate/redundant features
    select(-id, -funder, -wpt_name, -num_private, -subvillage, -lga, -ward, -recorded_by, -scheme_name) %>% # Remove Features that have too many levels or are not needed for the analysis
    mutate(year_recorded = as.numeric(format(date_recorded, "%Y"))) %>%
    mutate(month_recorded = as.numeric(format(date_recorded, "%m"))) %>%
    select(-date_recorded) %>%
    mutate(across(where(is.character), as.factor)) %>%
    mutate(permit = ifelse(permit, "yes", ifelse(!permit, "no", "unknown")), # Convert permit to factor with levels yes, no, unknown
           permit = ifelse(is.na(permit), "unknown", permit), # Replace NA values in permit with "unknown"
           public_meeting = ifelse(public_meeting, "yes", ifelse(!public_meeting, "no", "unknown")), # Convert public_meeting to factor with levels yes, no, unknown
           public_meeting = ifelse(is.na(public_meeting), "unknown", public_meeting), # Replace NA values in public_meeting with "unknown"
           construction_year = ifelse(construction_year == 0, median_construction_year, construction_year), # Replace missing values in construction_year with the median of the known values
           mean_longitude = ifelse(longitude != 0, longitude, mean_longitude), # Replace missing values in longitude with the mean of the known values
           mean_latitude = ifelse(latitude != 2e-8, latitude, mean_latitude), # Replace missing values in latitude with the mean of the known values
           installer = as.character(installer)) %>% # Convert installer to character
    mutate(installer = ifelse(installer %in% Top25_installers$installer, installer, "other")) %>% # Replace installers not in the top 25 with "other"
    select(-longitude, -latitude) %>% # Remove longitude and latitude, because they are replaced by mean_longitude and mean_latitude
    mutate(payment_type = factor(payment_type, ordered = TRUE, levels = c("per bucket", "monthly", "annually", "on failure", "other", "unknown", "never pay")), # Convert payment_type to factor with ordered levels
           permit = factor(permit), # Convert permit to factor with ordered levels
           public_meeting = factor(public_meeting), # Convert public_meeting to factor with ordered levels
           installer = as.factor(installer), # Convert installer to factor
           district_code = as.factor(district_code)) # Convert district_code to factor
