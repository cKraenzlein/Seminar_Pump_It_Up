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

summary(final_waterpoints)
nrow(final_waterpoints)
# --------------------------------------------------------------------------
# ward and lga
length(unique(final_waterpoints$ward))
length(unique(final_waterpoints$lga))
length(unique(final_waterpoints$subvillage))
category_counts <- table(final_waterpoints$ward)
category_percentages <- round(prop.table(category_counts) * 100, 2)

length(category_percentages[category_percentages > 1])

category_counts <- table(final_waterpoints$lga)
category_percentages <- round(prop.table(category_counts) * 100, 2)

length(category_percentages[category_percentages > 1])

category_counts <- table(final_waterpoints$subvillage)
category_percentages <- round(prop.table(category_counts) * 100, 2)

length(category_percentages[category_percentages > 1])
# --------------------------------------------------------------------------
# Table of all features that respresent the location of the waterpoint:
Location_data_unique <- final_waterpoints %>% 
    select(ward, lga, longitude, latitude, region, district_code, subvillage) %>%
    summarise(across(everything(), n_distinct)) %>% 
    pivot_longer(cols = everything(), names_to = "Feature", values_to = "Unique_Levels")

Location_data_NAs <- final_waterpoints %>% 
    select(ward, lga, longitude, latitude, region, district_code, subvillage) %>%
    summarise(across(everything(), ~sum(is.na(.)), .names = "NA_Count_{.col}")) %>% 
    pivot_longer(cols = everything(), names_to = "Feature", values_to = "NA_Count")

# Min Max from longitude and latitude
Location_data_min_max <- final_waterpoints %>% 
    select(longitude, latitude) %>%
    summarise(across(everything(), list(min = min, max = max), na.rm = TRUE)) 
# --------------------------------------------------------------------------
# funder
category_counts <- table(final_waterpoints$funder)
category_percentages <- round(prop.table(category_counts) * 100, 2)
category_percentages[category_percentages > 1] # Show only categories with more than 1% of the total observations
# --------------------------------------------------------------------------
# installer
category_counts <- table(final_waterpoints$installer)
category_percentages <- round(prop.table(category_counts) * 100, 2)
category_percentages[category_percentages > 1] # Show only categories with more than 1% of the total observations
# --------------------------------------------------------------------------
# payment
category_counts <- table(final_waterpoints$payment)
category_percentages <- round(prop.table(category_counts) * 100, 2)
category_percentages
# --------------------------------------------------------------------------
# extraction_type
category_counts <- table(final_waterpoints$extraction_type)
category_percentages <- round(prop.table(category_counts) * 100, 2)
category_percentages

# extraction_type_group
category_counts <- table(final_waterpoints$extraction_type_group)
category_percentages <- round(prop.table(category_counts) * 100, 2)
category_percentages

# extraction_type_class
category_counts <- table(final_waterpoints$extraction_type_class)
category_percentages <- round(prop.table(category_counts) * 100, 2)
category_percentages
# --------------------------------------------------------------------------
# water_quality
category_counts <- table(final_waterpoints$water_quality)
category_percentages <- round(prop.table(category_counts) * 100, 2)
category_percentages

# quality_group
category_counts <- table(final_waterpoints$quality_group)
category_percentages <- round(prop.table(category_counts) * 100, 2)
category_percentages
# --------------------------------------------------------------------------
# source
category_counts <- table(final_waterpoints$source)
category_percentages <- round(prop.table(category_counts) * 100, 2)
category_percentages

# source_type
category_counts <- table(final_waterpoints$source_type)
category_percentages <- round(prop.table(category_counts) * 100, 2)
category_percentages
# --------------------------------------------------------------------------
# waterpoint_type
category_counts <- table(final_waterpoints$waterpoint_type)
category_percentages <- round(prop.table(category_counts) * 100, 2)
category_percentages

# waterpoint_type_group
category_counts <- table(final_waterpoints$waterpoint_type_group)
category_percentages <- round(prop.table(category_counts) * 100, 2)
category_percentages
# --------------------------------------------------------------------------
# scheme_name
category_counts <- table(final_waterpoints$scheme_name)
category_percentages <- round(prop.table(category_counts) * 100, 2)
category_percentages

# scheme_management
category_counts <- table(final_waterpoints$scheme_management)
category_percentages <- round(prop.table(category_counts) * 100, 2)
category_percentages
# --------------------------------------------------------------------------
# Construction year
construction_year_data_1 <- final_waterpoints %>%
 filter(construction_year == 0) %>%
 select(date_recorded, construction_year) %>%
 summarise(count = n())

construction_year_data <- final_waterpoints %>%
 filter(construction_year > 0) %>%
 summarise(median_construction_year = median(construction_year, na.rm = TRUE))

# --------------------------------------------------------------------------
# Missing values
Data_Missing_values <- final_waterpoints %>%
    select(-payment_type, -region_code, -quantity_group) %>% # Remove all duplicate features
    select(-id, -num_private, -recorded_by, -wtp_name) %>% # Remove unique, almost unique and constant features
    select(-extraction_type, -extraction_type_group, -water_quality, -source, -waterpoint_type_group, -scheme_name) %>% # Remove features that appear more than ones, and have categories that contain less than 5% of the total observations, leave at least one feature
    mutate(across(where(is.character), as.factor)) %>% # Convert all character columns to factors
    mutate(district_code = as.factor(district_code)) %>% # Convert district_code  to factors
    mutate(installer = forcats::fct_lump(installer, prop = 0.01, other_level = "other"), 
           funder = forcats::fct_lump(funder, prop = 0.01, other_level = "other")) %>%
    mutate(extraction_type = forcats::fct_lump(extraction_type_class, prop = 0.03, other_level = "other"), 
           source_type = forcats::fct_lump(source_type, prop = 0.03, other_level = "other"), 
           waterpoint_type = forcats::fct_lump(waterpoint_type, prop = 0.03, other_level = "other"), 
           water_quality = forcats::fct_lump(quality_group, prop = 0.03, other_level = "other")) %>% 
    select(-extraction_type_class, -quality_group) %>% # Remove original columns
    mutate(year_recorded = as.numeric(format(date_recorded, "%Y")), # Extract year from date_recorded
           sin_month = sin(2 * pi * as.numeric(format(date_recorded, "%m")) / 12), # Extract month from date_recorded
           cos_month = cos(2 * pi * as.numeric(format(date_recorded, "%m")) / 12)) %>%
    select(-date_recorded) %>% # Remove original date_recorded column 




Long_Lat <- Data_Missing_values %>%
  group_by(longitude, latitude) %>%
  filter(n() > 1) %>%
  ungroup()

View(Long_Lat)

Duplicate_waterpoints <- Data_Missing_values %>%
    filter(longitude %in% Long_Lat$longitude & latitude %in% Long_Lat$latitude) %>%
    filter(!is.na(longitude) & !is.na(latitude))

Duplicate_waterpoints_same_date <- Data_Missing_values %>%
    group_by(longitude, latitude, date_recorded) %>%
    filter(n() > 1) %>%
    filter(!is.na(longitude) & !is.na(latitude)) %>%
    ungroup()
View(Duplicate_waterpoints_same_date)

View(Duplicate_waterpoints)

View(Data_Missing_values %>%
  # Group the data by the combination of longitude and latitude
  group_by(longitude, latitude) %>%
  # Keep only the groups that have more than one row
  filter(n() > 1) %>%
  # Select just the first row from each of the remaining groups
  slice(1) %>%
  # Remove the grouping
  ungroup())

nrow(Duplicate_waterpoints) / 2

summary(Data_Missing_values)
length(unique(Data_Missing_values$wpt_name))

sum(duplicated(Data_Missing_values$longitude)&duplicated(Data_Missing_values$latitude))
sum(is.na(Data_Missing_values$longitude))
length(unique(Data_Missing_values$longitude)) == nrow(Data_Missing_values)
library(data.table)
dt = data.table(Data_Missing_values)

dt[duplicated(longitude), cbind(.SD[1], number = .N), by = longitude]

missing <- final_waterpoints %>%
    filter(amount_tsh == 0 & population == 0 & gps_height == 0 & is.na(construction_year))

nrow(missing)


View(missing)
# --------------------------------------------------------------------------
library(VIM) # Visualize missing values
countNA(final_waterpoints) # Count missing values
missing_values_per_col <- aggr(final_waterpoints, plot = FALSE) # Count missing values per column
missing_values_per_col

# Get the number of rows in the data frame
n_obs <- nrow(final_waterpoints)

# Count missing values per column
missing_values_per_col <- aggr(final_waterpoints, plot = FALSE)$missings

# Calculate the percentage of missing values
missing_values_per_col$percentage <- (missing_values_per_col$Count / n_obs) * 100

# Display the extended table
missing_values_per_col %>%
    filter(percentage > 0)

histMiss(TrainingData, pos = 6, selection = "any")


par(mar = c(20, 4, 4, 6) + 0.1)
plot(missing_values_per_col, numbers = TRUE, cex.axis = 0.5, fig.pos="H", horiz = TRUE) # Plot the pattern of missing data
par(mar = c(5, 4, 4, 2) + 0.1)
mice::md.pattern(TrainingData)
# --------------------------------------------------------------------------
# Visualize missing data patterns from longitude and latitude:
#numeric features:
# amount_tsh:
marginplot(final_waterpoints[, c("amount_tsh", "longitude")],alpha = 0.65,pch = 20, cex = 1.3)
marginplot(final_waterpoints[, c("amount_tsh", "latitude")],alpha = 0.65,pch = 20, cex = 1.3)
# gps_height:
marginplot(final_waterpoints[, c("gps_height", "longitude")],alpha = 0.65,pch = 20, cex = 1.3)
marginplot(final_waterpoints[, c("gps_height", "latitude")],alpha = 0.65,pch = 20, cex = 1.3)
# population:
marginplot(final_waterpoints[, c("population", "longitude")],alpha = 0.65,pch = 20, cex = 1.3)
marginplot(final_waterpoints[, c("population", "latitude")],alpha = 0.65,pch = 20, cex = 1.3)
# construction_year:
marginplot(final_waterpoints[, c("construction_year", "longitude")],alpha = 0.65,pch = 20, cex = 1.3)
marginplot(final_waterpoints[, c("construction_year", "latitude")],alpha = 0.65,pch = 20, cex = 1.3)
# longitude x latitude:
marginplot(final_waterpoints[, c("longitude", "latitude")],alpha = 0.65,pch = 20, cex = 1.3)

# categorical features:
source("MissingDataVisualizationFunctions.r")
# lga: 
plot_missing_by_category(final_waterpoints, "latitude", "lga", TRUE)
plot_missing_by_category(final_waterpoints, "longitude", "lga", TRUE)
# region:
plot_missing_by_category(final_waterpoints, "latitude", "region")
plot_missing_by_category(final_waterpoints, "longitude", "region")
# quantity:
plot_missing_by_category(final_waterpoints, "latitude", "quantity")
plot_missing_by_category(final_waterpoints, "longitude", "quantity")
# funder:
plot_missing_by_category(final_waterpoints, "latitude", "funder", TRUE)
plot_missing_by_category(final_waterpoints, "longitude", "funder", TRUE)
# installer:
plot_missing_by_category(final_waterpoints, "latitude", "installer", TRUE)
plot_missing_by_category(final_waterpoints, "longitude", "installer", TRUE)
# scheme_management:
plot_missing_by_category(final_waterpoints, "latitude", "scheme_management")
plot_missing_by_category(final_waterpoints, "longitude", "scheme_management")
# payment:
plot_missing_by_category(final_waterpoints, "latitude", "payment")
plot_missing_by_category(final_waterpoints, "longitude", "payment")


final_waterpoints %>% 
    filter(gps_height == 0) %>% 
    nrow()
final_waterpoints %>% 
    filter(population == 0) %>% 
    nrow()
final_waterpoints %>% 
    filter(amount_tsh == 0) %>% 
    nrow()


spineMiss(TrainingData[, c("quantity", "construction_year")])

marginplot(TrainingData[, c("longitude", "construction_year")],
alpha = 0.65, # transparency of points
pch = 20, cex = 1.3)

marginplot(TrainingData[, c("latitude", "construction_year")],
alpha = 0.65, # transparency of points
pch = 20, cex = 1.3)

marginplot(TrainingData[, c("construction_year", "latitude")],
alpha = 0.65, # transparency of points
pch = 20, cex = 1.3)


marginplot(TrainingData[, c("year_recorded", "construction_year")],
alpha = 0.65, # transparency of points
pch = 20, cex = 1.3)

scattmatrixMiss(TrainingData[, c("quantity", "construction_year", "longitude", "latitude")], highlight = TRUE)

matrixplot(TrainingData, sortby = "date_recorded")


mosaicMiss(TrainingData, plotvars = c("status_group", "quantity"),
highlight = "public_meeting", miss.labels = FALSE)

library(skimr)
skim(TrainingData)






# 1. Separate the rows with NA coordinates
na_coords <- Data_Missing_values %>%
  filter(is.na(longitude) & is.na(latitude))

# Process the data to remove duplicates based on the lesser number of NAs
cleaned_waterpoints <- Data_Missing_values %>%
  filter(!is.na(longitude) & !is.na(latitude)) %>%
  # Group the data by the columns that define a duplicate
  group_by(longitude, latitude, as.numeric(format(date_recorded, "%Y")), as.numeric(format(date_recorded, "%m"))) %>%
  # Arrange the rows within each group. The row with the fewest NAs will be first.
  arrange(pick(everything()), rowSums(is.na(cur_data()))) %>%
  # Keep only the first row of each group, which is now the best one
  slice(1) %>%
  # Ungroup the data
  ungroup()

# 3. Combine the cleaned valid coordinates with the original NA rows
final_waterpoints <- bind_rows(cleaned_waterpoints, na_coords)

# Display the final data frame
nrow(final_waterpoints)
View(final_waterpoints)

Duplicate_waterpoints <- final_waterpoints %>%
    group_by(longitude, latitude) %>%
    filter(n() > 1) %>%
    filter(!is.na(longitude) & !is.na(latitude)) %>%
    ungroup()
View(Duplicate_waterpoints)
