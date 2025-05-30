# Load necessary packages
library(readr)
library(dplyr)
library(here)
# Set the working directory to the location of the script
here::here()
# Load the test data
TestData <- readr::read_csv("TestData.csv")

# Remov some variables that are not needed for the analysis and format the data, order some factors
TestData_factor_ordered <- TestData %>% 
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
           installer = as.factor(installer))

# Replace NA values in installer with "unknown"
TestData_factor_ordered$installer <- as.character(TestData_factor_ordered$installer)
TestData_factor_ordered$installer[is.na(TestData_factor_ordered$installer)] <- "unknown"
# Convert installer back to factor
TestData_factor_ordered$installer <- as.factor(TestData_factor_ordered$installer)

Id <- TestData$id



# Formatted Installer for Test Data
source("CreateTrainingData.r")

TestData_FormattedInstaller <- TestData_factor_ordered %>%
    mutate(installer = as.character(installer)) %>%
    mutate(installer = ifelse(installer %in% Top25_installers$installer, installer, "other")) %>%
    mutate(installer = as.factor(installer))
