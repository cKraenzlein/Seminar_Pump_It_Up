# Load necessary packages
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)
# Set the working directory to the location of the script
here::here()
# Read in the Datasets
TrainingData_Variables <- read_csv("TrainingData_Variables.csv")
TrainingData_TargetVariable <- read_csv("TrainingData_TargetVariable.csv")
# merge the data
TrainingData <- merge(x = TrainingData_Variables, y = TrainingData_TargetVariable, by = "id")
# 1. Separate the rows with NA coordinates
na_coords <- TrainingData %>%
  filter(is.na(longitude) & is.na(latitude))

# Process the data to remove duplicates based on the lesser number of NAs
cleaned_waterpoints <- TrainingData %>%
  filter(!is.na(longitude) & !is.na(latitude)) %>%
  mutate(year_recorded = as.numeric(format(date_recorded, "%Y")),
         month_recorded = as.numeric(format(date_recorded, "%m"))) %>%
  group_by(longitude, latitude, year_recorded, month_recorded) %>%
  # Arrange the rows within each group. The row with the fewest NAs will be first.
  arrange(pick(everything()), rowSums(is.na(cur_data()))) %>%
  # Keep only the first row of each group, which is now the best one
  slice(1) %>%
  # Ungroup the data
  ungroup() %>%
  select(-year_recorded, -month_recorded)
# 3. Combine the cleaned valid coordinates with the original NA rows
final_waterpoints <- bind_rows(cleaned_waterpoints, na_coords)


str(final_waterpoints)

data_plots <- final_waterpoints %>%
  mutate(flag = ifelse(population == 0 & amount_tsh == 0 & construction_year == 0 & gps_height == 0, "missing", "present"))

# Plot longitude and latitude with colored points based on the flag
ggplot(data_plots, aes(x = longitude, y = latitude, color = flag)) +
  geom_point() +
  labs(title = "Waterpoints by Location and Data Availability",
       x = "Longitude",
       y = "Latitude",
       color = "Data Availability") +
  theme_minimal()

# Remove wrong longitude and latitude values
data_plots_2 <- data_plots %>%
    mutate(longitude = ifelse(longitude == 0, NA, longitude),
           latitude = ifelse(latitude == -2e-8, NA, latitude))

# Plot longitude and latitude with colored gps_height
# Define your custom colors (in order of your legend)
col <- c("#08519c","#006400", "#228B22", "#ADFF2F", "#FFFF00", "#FFD700", "#FF8C00", "#FF0000")
col_breaks = c(-100, 0, 500, 1000, 1250, 1500, 2000, 2500, 5895)
col_labels = c("< -100", "0-500", "501-1000", "1001-1250", "1251-1500", "1501-2000", "2001-2500", "> 2501")

data_height <- data_plots_2 %>%
  filter(!is.na(longitude) & !is.na(latitude) & !is.na(gps_height)) %>%
  mutate(elevation = cut(gps_height,
                        breaks = col_breaks,
                        labels = col_labels,
                        right = FALSE))

location_elevation_with_missing <- ggplot(data_height, aes(x = longitude, y = latitude, color = elevation)) +
  geom_point(alpha = 0.7) +
  scale_color_manual(
    values = col) +
  labs(title = "Waterpoints by Location and GPS Height",
       x = "Longitude",
       y = "Latitude",
       color = "GPS Height")

data_height_without_missing <- data_height %>%
  select(-elevation) %>%
  mutate(gps_height = ifelse(flag == "missing", NA, gps_height)) %>%
  filter(!is.na(gps_height)) %>%
  filter(!is.na(longitude) & !is.na(latitude)) %>%
  mutate(elevation = cut(gps_height,
                        breaks = col_breaks,
                        labels = col_labels,
                        right = FALSE))

location_elevation_without_missing <- ggplot(data_height_without_missing, aes(x = longitude, y = latitude, color = elevation)) +
  geom_point(alpha = 0.7) +
  scale_color_manual(values = col) +
  labs(title = "Waterpoints by Location and GPS Height",
       x = "Longitude",
       y = "Latitude",
       color = "GPS Height")

nrow(data_height_without_missing)
nrow(data_height)
ggsave("location_elevation_without_missing.png", plot = location_elevation_without_missing, height = 5, width = 6, dpi = 800)
ggsave("location_elevation_with_missing.png", plot = location_elevation_with_missing, height = 5, width = 6, dpi = 800)

data_waterpoint_type <- data_plots_2%>%
  filter(!is.na(longitude) & !is.na(latitude) & !is.na(gps_height)) %>%
  select(longitude, latitude, waterpoint_type, gps_height, flag) %>%
  mutate(gps_height = ifelse(flag == "missing", NA, gps_height)) %>%
  mutate(waterpoint_type = factor(waterpoint_type))

location_waterpoint_type <- ggplot(data_waterpoint_type, aes(x = longitude, y = latitude, color = waterpoint_type)) +
  geom_point(alpha = 0.7) +
  labs(title = "Waterpoints by Location and Type",
       x = "Longitude",
       y = "Latitude",
       color = "Waterpoint Type")

distribution_gps_height_per_waterpoint_type <- ggplot(data_waterpoint_type, aes(x = waterpoint_type, y = gps_height)) +
  geom_boxplot() +
  labs(title = "Distribution of GPS Height by Waterpoint Type without Missing Values",
       x = "Waterpoint Type",
       y = "GPS Height")



# Query actual elevation values
#library(elevatr)
#library(sf)
#data_long_lat_for_missing_gps_height <- data_height %>%
#mutate(gps_height = ifelse(flag == "missing", NA, gps_height)) %>%
#  filter(is.na(gps_height)) %>%
#  select(longitude, latitude)
# Convert to sf object
#pts <- sf::st_as_sf(data_long_lat_for_missing_gps_height,
#                coords = c("longitude", "latitude"),
#                crs = 4326)  # WGS84
# Get elevations
#elev <- elevatr::get_elev_point(locations = pts, prj = "+proj=longlat +datum=WGS84")
#str(elev)
#head(elev)


# Log scaling the amount_tsh and population: 
data_log_transformed <- data_plots_2 %>%
  mutate(amount_tsh = ifelse(flag == "missing", NA, amount_tsh), 
         population = ifelse(flag == "missing", NA, population)) %>%
  filter(!is.na(amount_tsh) | !is.na(population)) %>%
  mutate(amount_tsh_log = log(amount_tsh + 1),  # Adding 1 to avoid log(0)
         population_log = log(population + 1))

# Plot longitude and latitude with colored population
location_population_without_missing <- ggplot(data_log_transformed, aes(x = longitude, y = latitude, color = population)) +
  geom_point(alpha = 0.7) +
  labs(title = "Waterpoints by Location and Population",
       x = "Longitude",
       y = "Latitude",
       color = "Population")

# Plot longitude and latitude with colored amount_tsh
location_amount_tsh_without_missing <- ggplot(data_log_transformed, aes(x = longitude, y = latitude, color = amount_tsh)) +
  geom_point(alpha = 0.7) +
  labs(title = "Waterpoints by Location and amount",
       x = "Longitude",
       y = "Latitude",
       color = "amount")

# Plot longitude and latitude with colored population_log
location_population_log_without_missing <- ggplot(data_log_transformed, aes(x = longitude, y = latitude, color = population_log)) +
  geom_point(alpha = 0.7) +
  labs(title = "Waterpoints by Location and Population (Log Scale)",
       x = "Longitude",
       y = "Latitude",
       color = "Population (Log Scale)")

# Plot longitude and latitude with colored amount_tsh_log
location_amount_tsh_log_without_missing <- ggplot(data_log_transformed, aes(x = longitude, y = latitude, color = amount_tsh_log)) +
  geom_point(alpha = 0.7) +
  labs(title = "Waterpoints by Location and amount (Log Scale)",
       x = "Longitude",
       y = "Latitude",
       color = "amount (Log Scale)")




#distribution of population: 
ggplot(data_log_transformed, aes(population)) +
  geom_density()

p1 <- ggplot(data_log_transformed, aes(x = longitude, y = amount_tsh_log)) +
  geom_point(alpha = 0.5)

p2 <- ggplot(data_log_transformed, aes(x = longitude, y = amount_tsh)) +
  geom_point(alpha = 0.5)

p1 + p2

p3 <- ggplot(data_log_transformed, aes(x = latitude, y = population_log)) +
  geom_point(alpha = 0.5)

p4 <- ggplot(data_log_transformed, aes(x = latitude, y = population)) +
  geom_point(alpha = 0.5)

p3 + p4
