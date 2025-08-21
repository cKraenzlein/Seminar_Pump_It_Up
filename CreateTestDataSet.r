# Load necessary packages
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
# Set the working directory to the location of the script
here::here()
# Load the test data
TestData <- readr::read_csv("TestData.csv")
Id <- TestData$id

Data_Test_Final <- TestData %>%
    select(-payment_type, -region_code, -quantity_group) %>% # Remove all duplicate features
    select(-id, -num_private, -recorded_by, -wpt_name) %>% # Remove unique, almost unique and constant features
    select(-extraction_type, -extraction_type_group, -water_quality, -source, -waterpoint_type_group, -scheme_name, -management_group, -management) %>% # Remove features that appear more than ones, and have categories that contain less than 5% of the total observations, leave at least one feature
    select(-subvillage, -ward, -lga) %>% 
    mutate(across(where(is.character), as.factor)) %>% # Convert all character columns to factors
    mutate(district_code = as.factor(district_code)) %>% # Convert district_code  to factors
    mutate(funder = forcats::fct_collapse(funder,
                unknown = c("0"), Dwsp = c("Dwsp"), Dwssp = c("Dwssp"), 
                Government_Of_Tanzania = c("Government Of Tanzania"), Hesawa = c("Hesawa"), Holland = c("Holland"), 
                Jica = c("Jica"), Lwi = c("Lwi"), Plan_International = c("Plan International"), 
                Ridep = c("Ridep"), Rwssp = c("Rwssp"), Tasaf = c("Tasaf"), 
                Unicef = c("Unicef"), World_Vision = c("World Vision"), Wsdp = c("Wsdp"), 
                Wvt = c("Wvt"), other_level = "DEPRECATED"), 
           installer = forcats::fct_collapse(installer,
                unknown = c("0"), DWE = c("DWE"), Government = c("Government"), Hesawa = "Hesawa",
                HOLLAND = c("HOLLAND"), JICA = c("JICA"), LWI = c("LWI"), Plan_Internationa = "Plan Internationa",
                RWE = c("RWE"), TASAF = c("TASAF"), World_Vision = c("World Vision"), WVT = "WVT",
                other_level = "DEPRECATED")) %>%
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
           scheme_management = forcats::fct_collapse(scheme_management, 
                other = c("Other", "None", "SWC", "Trust"), 
                Company = "Company",
                Parasental = "Parasental",
                Private_operator = "Private operator",
                VWC = "VWC",
                Water_authority = "Water authority",
                Water_Board = "Water Board",
                WUA = "WUA",
                WUG = "WUG")) %>% 
    select(-extraction_type_class, -quality_group) %>% # Remove original columns
    mutate(year_recorded = as.numeric(format(date_recorded, "%Y")), # Extract year from date_recorded
           sin_month = sin(2 * pi * as.numeric(format(date_recorded, "%m")) / 12), # Extract month from date_recorded
           cos_month = cos(2 * pi * as.numeric(format(date_recorded, "%m")) / 12)) %>%
    select(-date_recorded) %>% # Remove original date_recorded column 
    mutate(population_log = log(population + 1), # Adding 1 to avoid log(0)
           amount_tsh_log = log(amount_tsh + 1)) %>% # Adding 1 to avoid log(0)
    select(-population, -amount_tsh)