library(mlr3viz)
library(patchwork)
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





Plot_Tuning_RF <- function(instance) {

    numtrees <-     patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_randomForest.num.trees"))
    minnodesize <-  patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_randomForest.min.node.size"))
    mtry <-         patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_randomForest.mtry.ratio"))
    maxdepth <-     patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_randomForest.max.depth"))

    numtrees <- numtrees +
        labs(x = "number trees", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "none")

    minnodesize <- minnodesize +
        labs(x = "min node size", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "none")

    mtry <- mtry +
        labs(x = "mtry", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "none")
    maxdepth <- maxdepth +
        labs(x = "max depth", y = "Accuracy") +
        theme_bw()

    # Combine the plots
    Tuning_RF_combined <- numtrees + minnodesize + mtry + maxdepth +
        plot_layout(nrow = 2) +
        plot_annotation(title = "Random Forest Hyperparameter Tuning Results")

    # Save the plot
    ggsave("Tuning_RF.png", plot = Tuning_RF_combined, width = 8, height = 7, dpi = 400, path = "./Plots/HyperparameterTuning/")

    return(Tuning_RF_combined)
}

Plot_Tuning_LightGBM <- function(instance) {

    num_leaves <-               patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_lightGBM.num_leaves"))
    learning_rate <-            patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_lightGBM.learning_rate"))
    feature_fraction <-         patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_lightGBM.feature_fraction"))
    bagging_fraction <-         patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_lightGBM.bagging_fraction"))
    min_sum_hessian_in_leaf <-  patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_lightGBM.min_sum_hessian_in_leaf"))
    min_data_in_leaf <-         patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_lightGBM.min_data_in_leaf"))
    lambda_l1 <-                patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_lightGBM.lambda_l1"))
    lambda_l2 <-                patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_lightGBM.lambda_l2"))

    num_leaves <- num_leaves +
        labs(x = "number leaves", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "none")

    learning_rate <- learning_rate +
        labs(x = "learning rate", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "none")

    feature_fraction <- feature_fraction +
        labs(x = "feature fraction", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "none")

    bagging_fraction <- bagging_fraction +
        labs(x = "bagging fraction", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "none")

    min_sum_hessian_in_leaf <- min_sum_hessian_in_leaf +
        labs(x = "min sum hessian in leaf", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "none")

    min_data_in_leaf <- min_data_in_leaf +
        labs(x = "min data in leaf", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "none")

    lambda_l1 <- lambda_l1 +
        labs(x = "L1 regularization", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "none")

    lambda_l2 <- lambda_l2 +
        labs(x = "L2 regularization", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "bottom")

    # Combine the plots
    Tuning_LightGBM_combined <- num_leaves + learning_rate + feature_fraction + bagging_fraction + min_sum_hessian_in_leaf + min_data_in_leaf + lambda_l1 + lambda_l2 +
        plot_layout(nrow = 3) +
        plot_annotation(title = "LightGBM Hyperparameter Tuning Results")

    # Save the plot
    ggsave("Tuning_LightGBM.png", plot = Tuning_LightGBM_combined, width = 8, height = 9, dpi = 400, path = "./Plots/HyperparameterTuning/")

    return(Tuning_LightGBM_combined)
}

Plot_Tuning_XGBoost <- function(instance) {

    eta <-                  patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_classif.xgboost.eta"))
    max_depth <-            patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_classif.xgboost.max_depth"))
    nrounds <-              patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_classif.xgboost.nrounds"))
    min_child_weight <-     patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_classif.xgboost.min_child_weight"))
    subsample <-            patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_classif.xgboost.subsample"))
    colsample_bytree <-     patchwork::wrap_plots(autoplot(instance, type = "marginal", cols_x = "x_domain_classif.xgboost.colsample_bytree"))

    eta <- eta +
        labs(x = "Learning rate (eta)", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "none")

    max_depth <- max_depth +
        labs(x = "Maximum depth", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "none")

    nrounds <- nrounds +
        labs(x = "Number of rounds", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "none")

    min_child_weight <- min_child_weight +
        labs(x = "Minimum child weight", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "none")

    subsample <- subsample +
        labs(x = "Subsample ratio", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "none")

    colsample_bytree <- colsample_bytree +
        labs(x = "Colsample by tree", y = "Accuracy") +
        theme_bw() +
        theme(legend.position = "bottom")

    # Combine the plots into a single layout
    Tuning_XGBoost_combined <- eta + max_depth + nrounds + min_child_weight + subsample + colsample_bytree +
        plot_layout(nrow = 2) +
        plot_annotation(title = "XGBoost Hyperparameter Tuning Results")

    # Save the plot to file
    ggsave("Tuning_XGBoost.png", plot = Tuning_XGBoost_combined, width = 8, height = 7, dpi = 400, path = "./Plots/HyperparameterTuning/")

    return(Tuning_XGBoost_combined)
}
