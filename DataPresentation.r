#Pump it Up: Data Presentation
data <- read.csv("4910797b-ee55-40a7-8668-10efd5c1b960.csv")

str(data)
head(data)

table(data$water_quality)

library(tidyverse)
library(dplyr)

# Construction Year
data_plot_constructionyear <- data %>% select(construction_year) %>% filter(construction_year != 0)
ggplot2::ggplot(data_plot_constructionyear, aes(x = construction_year))+
    geom_bar()

quality <- data %>% select(quality_group)
table(quality)

# Water quality
data_plot_constructionyear_x_quality <- data %>% select(construction_year, quality_group) %>% filter(construction_year != 0)
ggplot2::ggplot(data_plot_constructionyear_x_quality, aes(x = construction_year, fill = quality_group))+
    geom_bar(position = "fill")


# GPS height
data_plot_GPS_height <- data %>% select(gps_height)
ggplot2::ggplot(data_plot_GPS_height, aes(y = gps_height))+
    geom_boxplot()

ggplot2::ggplot(data_plot_GPS_height, aes(x = gps_height))+
    geom_density()

# Region
data_plot_region <- data %>% select(region)
table(data_plot_region)

ggplot2::ggplot(data_plot_region, aes(x = region))+
    geom_bar()

# Region x Source
data_plot_region_x_source <- data %>% select(region, source)
ggplot2::ggplot(data_plot_region_x_source, aes(x = region, fill = source))+
    geom_bar(position = "fill")


data2 <- read.csv("0bf8bc6e-30d0-4c50-956a-603fc693d966.csv")
str(data2)
table(data2$status_group)

# Combine both Data Sets
data_combined <- merge(x = data, y = data2, by = "id")
str(data_combined)

# Construction Year / status_group
data_plot_constructionyear_status <- data_combined %>% select(construction_year, status_group) %>% filter(construction_year != 0)
ggplot2::ggplot(data_plot_constructionyear_status, aes(x = construction_year, fill = status_group))+
    geom_bar(position = "fill")

# Quantity 
data_plot_quatity <- data %>% select(quantity)
table(data_plot_quatity)

# Quantity x Status
data_plot_quatity_x_status <- data_combined %>% select(quantity_group, status_group)
ggplot2::ggplot(data_plot_quatity_x_status, aes(x = quantity_group, fill = status_group))+
    geom_bar(position = "fill")

# Amount when not 0
data_plot_amount <- data %>% select(amount_tsh) # %>% filter(amount_tsh != 0)
ggplot2::ggplot(data_plot_amount, aes(x = amount_tsh))+
    geom_density()

mean(data_plot_amount$amount_tsh)
min(data_plot_amount$amount_tsh)
max(data_plot_amount$amount_tsh)
