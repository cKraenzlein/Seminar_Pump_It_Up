# Load necessary packages
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
# Set the working directory to the location of the script
here::here()
# Read in the Datasets
TrainingData_Variables <- read_csv("TrainingData_Variables.csv")
TrainingData_TargetVariable <- read_csv("TrainingData_TargetVariable.csv")
# Combine the two datasets by the column "id"
TrainingData <- merge(x = TrainingData_Variables, y = TrainingData_TargetVariable, by = "id") %>% 
    mutate(flag = ifelse(population == 0 & amount_tsh == 0 & construction_year == 0 & gps_height == 0, TRUE, FALSE)) %>%
    mutate(population = ifelse(flag, NA, population),
           amount_tsh = ifelse(flag, NA, amount_tsh),
           gps_height = ifelse(flag, NA, gps_height),
           construction_year = ifelse(flag, NA, construction_year)) %>%
    select(-flag) %>%
    mutate(construction_year = ifelse(construction_year == 0, NA, construction_year), 
           longitude = ifelse(longitude == 0, NA, longitude),
           latitude = ifelse(latitude == -2e-8, NA, latitude))
# 1. Separate the rows with NA coordinates
na_coords <- TrainingData %>%
  filter(is.na(longitude) & is.na(latitude))
# Process the data to remove duplicates based on the lesser number of NAs
cleaned_waterpoints <- TrainingData %>%
  filter(!is.na(longitude) & !is.na(latitude)) %>%
  mutate(year_recorded = as.numeric(format(date_recorded, "%Y")),
         month_recorded = as.numeric(format(date_recorded, "%m"))) %>%
  group_by(longitude, latitude, year_recorded, month_recorded) %>%
  arrange(pick(everything()), rowSums(is.na(cur_data()))) %>%
  slice(1) %>%
  ungroup() %>%
  select(-year_recorded, -month_recorded)
# 3. Combine the cleaned valid coordinates with the original NA rows
final_waterpoints <- bind_rows(cleaned_waterpoints, na_coords)

Data_Training_Final <- final_waterpoints %>%
    select(-payment_type, -region_code, -quantity_group) %>% # Remove all duplicate features
    select(-id, -num_private, -recorded_by, -wpt_name) %>% # Remove unique, almost unique and constant features
    select(-extraction_type, -extraction_type_group, -water_quality, -source, -waterpoint_type_group, -scheme_name, -management_group) %>% # Remove features that appear more than ones, and have categories that contain less than 5% of the total observations, leave at least one feature
    select(-subvillage, -ward, -lga) %>% 
    mutate(across(where(is.character), as.factor)) %>% # Convert all character columns to factors
    mutate(district_code = as.factor(district_code)) %>% # Convert district_code  to factors
    mutate(installer = forcats::fct_lump(installer, prop = 0.01, other_level = "other"), 
           funder = forcats::fct_lump(funder, prop = 0.01, other_level = "other")) %>%
    mutate(extraction_type = forcats::fct_lump(extraction_type_class, prop = 0.01, other_level = "other"), 
           source_type = forcats::fct_lump(source_type, prop = 0.01, other_level = "other"), 
           waterpoint_type = forcats::fct_lump(waterpoint_type, prop = 0.01, other_level = "other"), 
           water_quality = forcats::fct_lump(quality_group, prop = 0.01, other_level = "other"), 
           scheme_management = forcats::fct_lump(scheme_management, prop = 0.01, other_level = "other"), 
           management = forcats::fct_lump(management, prop = 0.01, other_level = "other")) %>% 
    select(-extraction_type_class, -quality_group) %>% # Remove original columns
    mutate(year_recorded = as.numeric(format(date_recorded, "%Y")), # Extract year from date_recorded
           sin_month = sin(2 * pi * as.numeric(format(date_recorded, "%m")) / 12), # Extract month from date_recorded
           cos_month = cos(2 * pi * as.numeric(format(date_recorded, "%m")) / 12)) %>%
    select(-date_recorded) %>% # Remove original date_recorded column 
    mutate(population_log = log(population + 1), # Adding 1 to avoid log(0)
           amount_tsh_log = log(amount_tsh + 1)) %>% # Adding 1 to avoid log(0)
    select(-population, -amount_tsh)

summary(Data_Training_Final)

library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(randomForest)

task <- as_task_classif(Data_Training_Final, target = "status_group")
learner <- lrn("regr.ranger")

task$missings()

mlr3viz::autoplot(task$clone()$select(c("amount_tsh_log", "gps_height")), type = "pairs")

#imp_missind = po("missind")

imp_num = po("imputelearner", learner = lrn("regr.ranger"), affect_columns = selector_type("numeric"), id = "impute_num")
imp_factor <- po("imputelearner", learner = lrn("regr.ranger"), affect_columns = selector_type("factor"), id = "impute_factor")
imp_bin <- po("imputelearner", learner = lrn("regr.ranger"), affect_columns = selector_type("logical"), id = "impute_bin")

imp_all <- imp_num %>>% imp_factor %>>% imp_bin

#task_ext = imp_missind$train(list(task))[[1]]
#task_ext$data()

task_ext = imp_all$train(task)
imputed_task <- task_ext[[1]]
imputed_training_data <- imputed_task$data()

View(imputed_training_data)

# Save the Data in a CSV file
write.csv(imputed_training_data, "imputed_training_data.csv", row.names = FALSE)

# Load the test data
TestData <- readr::read_csv("TestData.csv")
Id <- TestData$id

Data_Test_Final <- TestData %>%
    select(-payment_type, -region_code, -quantity_group) %>% # Remove all duplicate features
    select(-id, -num_private, -recorded_by, -wpt_name) %>% # Remove unique, almost unique and constant features
    select(-extraction_type, -extraction_type_group, -water_quality, -source, -waterpoint_type_group, -scheme_name, -management_group) %>% # Remove features that appear more than ones, and have categories that contain less than 5% of the total observations, leave at least one feature
    select(-subvillage, -ward, -lga) %>% 
    mutate(across(where(is.character), as.factor)) %>% # Convert all character columns to factors
    mutate(district_code = as.factor(district_code)) %>% # Convert district_code  to factors
    mutate(installer = forcats::fct_collapse(installer,
                unknown = "0", Danida = "Danida", Dhv = "Dhv", 
                District_Council = "District Council", Dwsp = "Dwsp", Germany_Republi = "Germany Republi",
                Government_Of_Tanzania = "Government Of Tanzania", Hesawa = "Hesawa", Kkkt = "Kkkt", 
                Ministry_Of_Water = "Ministry Of Water", Norad = "Norad", Private_Individual = "Private Individual",
                Rwssp = "Rwssp", Tasaf = "Tasaf", Tcrs = "Tcrs",
                Unicef = "Unicef", Water = "Water", World_Bank = "World Bank",
                World_Vision = "World Vision"), 
           funder = forcats::fct_collapse(funder,
                unknown = "0", Central_Government = "Central government", CES = "CES",
                Commu = "Commu", DANIDA = "DANIDA", DWE = "DWE",
                Government = "Government", Hesawa = "Hesawa", KKKT = "KKKT",
                RWE = "RWE", TCRS = "TCRS")) %>%
    mutate(extraction_type = forcats::fct_collapse(extraction_type_class, 
                other = c("other", "wind-powered", "rope pump"), 
                gravity = "gravity",
                handpump = "handpump", 
                motorpump = "motorpump", 
                submersible = "submersible"), 
           waterpoint_type = forcats::fct_collapse(waterpoint_type, 
                other = c("other", "cattle trough", "dam"), 
                communal_standpipe = "communal standpipe", 
                improved_spring = "improved spring",
                communal_standpipe_multiple = "communal standpipe multiple"), 
            water_quality = forcats::fct_collapse(quality_group, 
                other = c("colored", "fluoride"), 
                good = "good", 
                salty = "salty",
                milky = "milky",
                unknown = "unknown"), 
           scheme_management = forcats::fct_lump(scheme_management, prop = 0.01, other_level = "other"), 
           management = forcats::fct_lump(management, prop = 0.01, other_level = "other")) %>% 
    select(-extraction_type_class, -quality_group) %>% # Remove original columns
    mutate(year_recorded = as.numeric(format(date_recorded, "%Y")), # Extract year from date_recorded
           sin_month = sin(2 * pi * as.numeric(format(date_recorded, "%m")) / 12), # Extract month from date_recorded
           cos_month = cos(2 * pi * as.numeric(format(date_recorded, "%m")) / 12)) %>%
    select(-date_recorded) %>% # Remove original date_recorded column 
    mutate(population_log = log(population + 1), # Adding 1 to avoid log(0)
           amount_tsh_log = log(amount_tsh + 1)) %>% # Adding 1 to avoid log(0)
    select(-population, -amount_tsh)

# Visualization
# Plot longitude and latitude with colored gps_height
# Define your custom colors (in order of your legend)
col <- c("#08519c","#006400", "#228B22", "#ADFF2F", "#FFFF00", "#FFD700", "#FF8C00", "#FF0000")
col_breaks = c(-100, 0, 500, 1000, 1250, 1500, 2000, 2500, 5895)
col_labels = c("< -100", "0-500", "501-1000", "1001-1250", "1251-1500", "1501-2000", "2001-2500", "> 2501")

data_height <- Data_Training_imputed %>%
  mutate(elevation = cut(gps_height,
                        breaks = col_breaks,
                        labels = col_labels,
                        right = FALSE))

location_elevation_with_imputed <- ggplot(data_height, aes(x = longitude, y = latitude, color = elevation)) +
  geom_point(alpha = 0.7) +
  scale_color_manual(
    values = col) +
  labs(title = "Waterpoints by Location and GPS Height with Imputed Values",
       x = "Longitude",
       y = "Latitude",
       color = "GPS Height")

ggsave("location_elevation_with_imputed.png", plot = location_elevation_with_imputed, height = 5, width = 6, dpi = 400)

task_ext$missings()
