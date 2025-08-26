library(ggplot2)

plot_importance <- function(importance_score, model) {
    importance_dt = data.table(
        Feature = names(importance_score),
        Importance = as.numeric(importance_score))
    # Visualize
    plot_importance = ggplot(importance_dt, aes(x = reorder(Feature, Importance), y = Importance)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        coord_flip() + # Flip coordinates to make it a horizontal bar plot
        labs(x = "Feature",
             y = "Importance") +
        theme_bw()

    Filename <- paste0("Importance_", model, ".png")
    ggsave(Filename, plot = plot_importance, width = 8, height = 5, dpi = 400, path = "./Plots/Importance/")

    return(plot_importance)
}