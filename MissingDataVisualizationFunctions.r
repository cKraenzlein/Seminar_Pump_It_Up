library(dplyr)
library(ggplot2)

# Function to calculate and plot the proportion of missing values in a numeric feature
plot_missing_by_category <- function(data, numeric_feature, categorical_feature, remove_zero = FALSE) {
  
  # Create a column indicating whether the numeric feature is missing
  missing_data <- data %>%
    mutate(missing = is.na(.data[[numeric_feature]])) %>%
    # Group by the categorical feature and calculate the proportion of missing data
    group_by(.data[[categorical_feature]]) %>%
    summarise(proportion_missing = mean(missing, na.rm = TRUE)) %>%
    # Remove categories with 0% missing data if remove_zero is TRUE
    filter(!(remove_zero & proportion_missing == 0)) %>%
    # Arrange by proportion of missing data in descending order
    arrange(desc(proportion_missing))
  
  # Plot the results
  ggplot(missing_data, aes(x = reorder(.data[[categorical_feature]], -proportion_missing), 
                           y = proportion_missing)) +
    geom_bar(stat = "identity", fill = "red") +
    labs(
      title = paste("Proportion of Missing", numeric_feature, "by", categorical_feature),
      x = categorical_feature,
      y = paste("Proportion of Missing", numeric_feature)
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
}