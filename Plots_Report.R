# Load necessary packages
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)
library(VIM)
# Set the working directory
here::here()
# Read in the Training Data
source("./Data/CreateTrainingData.r")
Data_Plots <- Data_Training_Final
# Create additional columns that specify if a feature is missing in that column
Data_Plots_flagges <- Data_Plots %>%
  mutate(Missing_amount_population_gps = factor(ifelse(is.na(population_log), "missing", "available"), levels = c("missing", "available")), 
         Missing_construction_year = factor(ifelse(is.na(construction_year), "missing", "available"), levels = c("missing", "available")), 
         Missing_lon_lat = factor(ifelse(is.na(longitude), "missing", "available"), levels = c("missing", "available")), 
         Missing_funder = factor(ifelse(is.na(funder), "missing", "available"), levels = c("missing", "available")), 
         Missing_installer = factor(ifelse(is.na(installer), "missing", "available"), levels = c("missing", "available")), 
         Missing_scheme_management = factor(ifelse(is.na(scheme_management), "missing", "available"), levels = c("missing", "available")), 
         Missing_public_meeting = factor(ifelse(is.na(public_meeting), "missing", "available"), levels = c("missing", "available")), 
         Missing_permit = factor(ifelse(is.na(permit), "missing", "available"), levels = c("missing", "available")))

Plot_location_missing_1 <- ggplot(Data_Plots_flagges, aes(x = longitude, y = latitude, col = Missing_amount_population_gps)) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c("#5D3A66", "#5AA9A5")) + 
  theme_bw() +
  theme(legend.position = "top", 
        legend.title = element_text(size = 20), 
        legend.text = element_text(size = 20), 
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16)) +
  labs(color = "Data:", x = "Longitude", y = "Latitude")  +
  guides(color = guide_legend(override.aes = list(size = 5)))
  


# Combined Plot of all categorical and logical features, Missingness by location
plot_location_installer <- ggplot(Data_Plots_flagges, aes(x = longitude, y = latitude, col = Missing_installer)) +
  geom_point(alpha = 0.5) + 
  scale_color_manual(values = c(available = "#5D3A66", missing = "#5AA9A5")) +
  theme_bw() +
  theme(legend.position = "none", 
        plot.title = element_text(size=22),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16)) +
  labs(title = "Installer")

plot_loction_funder <- ggplot(Data_Plots_flagges, aes(x = longitude, y = latitude, col = Missing_funder)) +
  geom_point(alpha = 0.5) + 
  scale_color_manual(values = c(available = "#5D3A66", missing = "#5AA9A5")) +
  theme_bw() +
  theme(legend.position = "none",
        plot.title = element_text(size=22),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16)) +
  labs(title = "Funder")

plot_location_scheme <- ggplot(Data_Plots_flagges, aes(x = longitude, y = latitude, col = Missing_scheme_management)) +
  geom_point(alpha = 0.5) + 
  scale_color_manual(values = c(available = "#5D3A66", missing = "#5AA9A5"))+ 
  theme_bw() +
  theme(legend.position = "none", 
        plot.title = element_text(size=22),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16)) +
  labs(title = "Scheme management")

plot_location_public <- ggplot(Data_Plots_flagges, aes(x = longitude, y = latitude, col = Missing_public_meeting)) +
  geom_point(alpha = 0.5) + 
  scale_color_manual(values = c(available = "#5D3A66", missing = "#5AA9A5")) +
  theme_bw() +
  theme(legend.position = "none", 
        plot.title = element_text(size=22),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16)) +
  labs(title = "Public meetings")

plot_location_permit <- ggplot(Data_Plots_flagges, aes(x = longitude, y = latitude, col = Missing_permit)) +
  geom_point(alpha = 0.5) + 
  scale_color_manual(values = c(available = "#5D3A66", missing = "#5AA9A5")) +
  theme_bw() +
  theme(legend.position = "bottom", 
        plot.title = element_text(size=22),
        legend.title = element_text(size = 20), 
        legend.text = element_text(size = 20), 
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16)) +
  labs(title = "Permit", color = "Data:")  +
  guides(color = guide_legend(override.aes = list(size = 5)))


plot_comined_missing_factor <- plot_location_installer + plot_loction_funder + plot_location_scheme + plot_location_public + plot_location_permit +
  nrow(2) +  plot_layout(axis_titles = "collect")

# Visualization of the log scaled features amount_tsh and population
Data_Plot_log <- Data_Plots_flagges %>%
  filter(!is.na(gps_height)) %>%
  select(population_log, amount_tsh_log)

Plot_log_amount <- ggplot(Data_Plot_log, aes(x = amount_tsh_log)) + 
  geom_boxplot(fill = "#3A6A97") +
  theme_bw() +
  theme(plot.title = element_text(size=22),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_blank(),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_blank()) +
  labs(title = "logscaled amount", x = "log(amount + 1)")

Plot_log_population <- ggplot(Data_Plot_log, aes(x = population_log)) + 
  geom_boxplot(fill = "#3A6A97") +
  theme_bw() +
  theme(plot.title = element_text(size=22),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_blank(),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_blank()) +
  labs(title = "logscaled population", x = "log(population + 1)")

plot_log_combined <- Plot_log_population / Plot_log_amount +
  nrow(2)

# Plot longitude and latitude with colored gps_height
# Define your custom colors (in order of your legend)
col_2 <- c("#08519C","#B0E0E6","#006400","#228B22","#ADFF2F", "#FFFF00","#FFD700","#FF8C00","#FF0000" )
col_breaks = c(-100, 0, 1, 500, 1000, 1250, 1500, 2000, 2500, 5895)
col_labels = c("< 0", "0", "1-500", "501-1000", "1001-1250", "1251-1500", "1501-2000", "2001-2500", "> 2501")

data_height_plot <- Data_Plots_flagges %>%
  filter(!is.na(longitude) & !is.na(latitude)) %>%
  mutate(gps_height = ifelse(is.na(gps_height), 0, gps_height)) %>%
  mutate(elevation = cut(gps_height,
                         breaks = col_breaks,
                         labels = col_labels,
                         right = FALSE))

location_elevation_with_missing <- ggplot(data_height_plot, aes(x = longitude, y = latitude, color = elevation)) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = col_2) +
  theme_bw() + 
  theme(legend.title = element_text(size = 20), 
        legend.text = element_text(size = 20), 
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16)) +
  labs(x = "Longitude",
       y = "Latitude",
       color = "GPS Height")  +
  guides(color = guide_legend(override.aes = list(size = 5)))

# Correlations in our data set:
#task = as_task_classif(na.omit(Data), target = "status_group", id = "PumpItUp")
#task2 <- task$clone()$select(setdiff(task$feature_names, c("district_code", "funder", "region")))
#mlr3viz::autoplot(task2, type = "pairs")

# Distribution of the target status_group
target_plot <- ggplot(Data_Plots_flagges, aes(x = status_group)) +
  geom_bar(fill = "#3A6A97") +
  geom_text(stat = "count", aes(label = scales::percent(after_stat(count)/ sum(after_stat(count)), accuracy = 0.1),
                                y = (after_stat(count))), vjust = -0.15, size = 6) +
  labs(x = "Status of the Waterpoint",
       y = "Count") +
  theme_bw() + 
  theme(axis.text = element_text(size = 15, color = "black"),
        axis.title = element_text(size = 17, color = "black", face = "bold"))
# Distribution of quatity
quantity_plot <- ggplot(Data_Plots_flagges, aes(x = quantity)) +
  geom_bar(fill = "#3A6A97") +
  geom_text(stat = "count", aes(label = scales::percent(after_stat(count)/ sum(after_stat(count)), accuracy = 0.1),
                                y = (after_stat(count))), vjust = -0.15, size = 6)  +
  labs(x = "Quantity",
       y = "Count") +
  theme_bw() + 
  theme(axis.text = element_text(size = 15, color = "black"), 
        axis.title = element_text(size = 17, color = "black", face = "bold"))
# Distrobution of basin
basin_plot <- ggplot(Data_Plots_flagges, aes(x = basin)) +
  geom_bar(fill = "#3A6A97") +
  geom_text(stat = "count", aes(label = scales::percent(after_stat(count)/ sum(after_stat(count)), accuracy = 0.1),
                                y = (after_stat(count))), vjust = -0.15, size = 6)  +
  labs(x = "Quantity",
       y = "Count") +
  theme_bw() + 
  theme(axis.text = element_text(size = 15, color = "black"), 
        axis.title = element_text(size = 17, color = "black", face = "bold"), 
        axis.text.x = element_text(angle = 45, hjust = 1))

# Combine 
# Create a column indicating whether the numeric feature is missing
missing_data <- Data_Plots %>%
  mutate(missing = is.na(longitude)) %>%
  # Group by the categorical feature and calculate the proportion of missing data
  group_by(region) %>%
  summarise(proportion_missing = mean(missing, na.rm = TRUE)) %>%
  # Remove categories with 0% missing data if remove_zero is TRUE
  filter(!(proportion_missing == 0)) %>%
  # Arrange by proportion of missing data in descending order
  arrange(desc(proportion_missing))

# Plot the results
Plot_long_lat_missing_region <- ggplot(missing_data, aes(x = reorder(region, -proportion_missing), 
                         y = proportion_missing)) +
  geom_bar(stat = "identity", fill = "#5AA9A5") +
  labs(
    x = "Region",
    y = paste("Percentage Missing Data")
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1), 
        axis.text = element_text(size = 15), 
        axis.title = element_text(size = 17))

# Vim Package
library(VIM) # Visualize missing values
countNA(Data_Plots) # Count missing values
missing_values_per_col <- aggr(Data_Plots, plot = FALSE) # Count missing values per column
missing_values_per_col

# Get the number of rows in the data frame
n_obs <- nrow(Data_Plots)

# Count missing values per column
missing_values_per_col <- aggr(Data_Plots, plot = FALSE)$missings

# Calculate the percentage of missing values
missing_values_per_col$percentage <- (missing_values_per_col$Count / n_obs) * 100

# Display the extended table
missing_values_per_col %>%
  filter(percentage > 0) %>%
  arrange(desc(Count))

# Visualize missing data patterns from longitude and latitude:
#numeric features:
# amount_tsh:
marginplot(Data_Plots[, c("amount_tsh_log", "longitude")],alpha = 0.65,pch = 20, cex = 1.3)
marginplot(Data_Plots[, c("amount_tsh_log", "latitude")],alpha = 0.65,pch = 20, cex = 1.3)
# gps_height:
marginplot(Data_Plots[, c("gps_height", "longitude")],alpha = 0.65,pch = 20, cex = 1.3)
marginplot(Data_Plots[, c("gps_height", "latitude")],alpha = 0.65,pch = 20, cex = 1.3)
# population:
marginplot(Data_Plots[, c("population_log", "longitude")],alpha = 0.65,pch = 20, cex = 1.3)
marginplot(Data_Plots[, c("population_log", "latitude")],alpha = 0.65,pch = 20, cex = 1.3)
# construction_year:
marginplot(Data_Plots[, c("construction_year", "longitude")],alpha = 0.65,pch = 20, cex = 1.3)
marginplot(Data_Plots[, c("construction_year", "latitude")],alpha = 0.65,pch = 20, cex = 1.3)

barMiss(Data_Plots[, c("region", "funder")])

a <- aggr(Data_Plots, plot = FALSE)
par(mar = c(20, 4, 4, 6) + 0.1)
plot(a, numbers = TRUE, fig.pos = "H")
par(mar = c(5, 4, 4, 2) + 0.1)


