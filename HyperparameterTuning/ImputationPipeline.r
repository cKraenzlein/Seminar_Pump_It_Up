# Imputation Pipeline
RF_imputation_regr = lrn("regr.ranger",num.threads = 8, num.trees = 250,min.node.size = 5,mtry = 6,max.depth=10)
RF_imputation_classif = lrn("classif.ranger",num.threads = 8, num.trees = 250,min.node.size = 5,mtry = 6,max.depth=10)

flag_missing = po("missind", affect_columns = selector_type(c("numeric", "logical", "factor")))
imp_num = po("imputelearner", learner = RF_imputation_regr, affect_columns = selector_name(c("longitude", "latitude")), id = "impute_num")
imp_factor = po("imputelearner", learner = RF_imputation_classif, affect_columns = selector_type("factor"), id = "impute_factor")
imp_bin = po("imputelearner", learner = RF_imputation_classif, affect_columns = selector_type("logical"), id = "impute_bin")
po_select = po("select", selector = selector_invert(selector_name(c("missing_population_log", "missing_gps_height", "missing_longitude", "missing_construction_year", "missing_amount_tsh_log"))))

imp_sel <- gunion(list(flag_missing, imp_num, imp_factor, imp_bin)) %>>% po("featureunion") %>>% po_select
