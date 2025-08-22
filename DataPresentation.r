#Pump it Up: Data Presentation
#data <- read.csv("4910797b-ee55-40a7-8668-10efd5c1b960.csv")
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


source("CreateTrainingData.r")
TrainingData$status_group

source("CreateTrainingData.r")
Data <- TrainingData_factor_ordered


Top25_installers <- Data %>%
    select(installer) %>%
    group_by(installer) %>%
    summarise(count = n()) %>%
    arrange(desc(count)) %>%
    head(25)

sum(Top25_installers$count)

Top100_installers <- Data %>%
    select(installer) %>%
    group_by(installer) %>%
    summarise(count = n()) %>%
    arrange(desc(count)) %>%
    head(100)

sum(Top100_installers$count)


# Installer x Status
data_plot_1 <- Data %>% select(installer, status_group)

ggplot2::ggplot(data_plot_1, aes(x = installer, fill = status_group)) +
    geom_bar(position = "fill")


source("CreateTrainingData.r")
Data_Plot <- TrainingData_FormattedInstaller %>%
    select(installer, status_group)

ggplot2::ggplot(Data_Plot, aes(x = installer, fill = status_group)) +
    geom_bar(position = "fill")

Data_PlotII <- TrainingData_factor_ordered

installer_count <- Data_PlotII %>%
    group_by(installer) %>%
    summarise(count = n())

Over_1000_installers <- installer_count %>%
    filter(count > 1000)

Over_500_installers <- installer_count %>%
    filter(count > 500 & count <= 1000)

Over_200_installers <- installer_count %>%
    filter(count > 200 & count <= 500)

Over_100_installers <- installer_count %>%
    filter(count > 100 & count <= 200)

Over_50_installers <- installer_count %>%
    filter(count > 50 & count <= 100)

Over_25_installers <- installer_count %>%
    filter(count > 25 & count <= 50)

Over_0_installers <- installer_count %>%
    filter(count > 0 & count <= 25)

Data_PlotII <- Data_PlotII %>%
    mutate(installer = as.character(installer)) %>%
    mutate(installer = ifelse(installer %in% Over_1000_installers$installer, "Over_1000_wells", 
                            ifelse(installer %in% Over_500_installers$installer, "Over_500_wells", 
                                    ifelse(installer %in% Over_200_installers, "Over_200_wells", 
                                            ifelse(installer %in% Over_100_installers$installer, "Over_100_wells", 
                                                    ifelse(installer %in% Over_50_installers$installer, "Over_50_wells", 
                                                            ifelse(installer %in% Over_25_installers$installer, "Over_25_wells", "under_25"))))))) %>%
    mutate(installer = as.factor(installer))

ggplot2::ggplot(Data_PlotII, aes(x = installer, fill = status_group)) +
    geom_bar(position = "fill")



data_plot_1 %>%
    group_by(installer) %>%
    summarise(
        count = n(), 
        percentage = (n() / nrow(data_plot_1))) %>%
    ungroup() %>%
    head(10)



is.null(data_plot_1)
class(data_plot_1)

dim(data_plot_1)
head(data_plot_1)
