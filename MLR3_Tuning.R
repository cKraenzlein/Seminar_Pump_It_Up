

# 0) Pakete laden  
suppressPackageStartupMessages({  
  library(mlr3)  
  library(mlr3learners)  
  library(mlr3tuning)  
  library(mlr3mbo)  
  library(paradox)  
  library(data.table)  
  library(future)  
  library(lgr)  
})  

# 1) Logging aktivieren  
lgr::get_logger("bbotk")$set_threshold("info")  
lgr::get_logger("mlr3")$set_threshold("info")  

# 2) Daten & Feature‑Selection  
dense_train <- as.matrix(x_train[, best_set])  
train_df    <- as.data.frame(dense_train)  
train_df$status_group <- as.factor(y_train)  
train_dt    <- as.data.table(train_df)  

# 3) Task definieren  
task <- TaskClassif$new("pump_fs", backend = train_dt, target = "status_group")  

# 4) Learner konfigurieren  
learner <- lrn("classif.xgboost",  
               predict_type = "prob",  
               eval_metric  = "merror")  

# 5) Reduzierter Hyperparameter‑Suchraum  
param_space <- ps(  
  nrounds          = p_int(500, 1500),  
  eta              = p_dbl(0.01, 0.3),  
  max_depth        = p_int(3, 12),  
  min_child_weight = p_int(1, 10),  
  subsample        = p_dbl(0.6, 1)  
)  

# 6) Resampling & Metrik  
resampling <- rsmp("repeated_cv", folds = 3, repeats = 1)  
measure    <- msr("classif.acc")  

# 7) Terminator: 2 Stunden  
terminator <- trm("run_time", secs = 1 * 3600)  

# 8) Tuning‑Instanz  
instance <- TuningInstanceBatchSingleCrit$new(  
  task         = task,  
  learner      = learner,  
  resampling   = resampling,  
  measure      = measure,  
  search_space = param_space,  
  terminator   = terminator  
)  

# 9) MBO‑Control: kleinstes Initial‑Design (5 Punkte)  
# 9) Tuner: Bayesian Optimization (mit Default-Initial-Design)
tuner <- tnr("mbo")

# 10) Parallelisierung
future::plan("multisession")

# 11) Tuning‑Loop
# 1) Prüfen, dass Initial‑Design schon da ist
# … vorheriges Setup bleibt unverändert …











# Vorausgesetzt: `instance`, `tuner` (tnr("mbo")), `future::plan("multisession")`
# und Logging sind bereits gesetzt, sowie dein `instance$archive` mit Initial-Design.

# Lege die maximale Anzahl an Batches fest
max_batches <- 50L

for (batch_id in seq_len(max_batches)) {
  message(sprintf("🔁 Starte Batch %02d – %s", batch_id, format(Sys.time(), "%H:%M:%S")))
  
  # Neue MBO‑Iteration
  tuner$optimize(instance)
  
  # Nur atomic Spalten für CSV-Export auswählen
  df_clean <- instance$archive$data[
    , setdiff(names(instance$archive$data),
              c("learner_param_vals", "x_domain")),
    with = FALSE
  ]
  
  # Ergebnisse als CSV speichern
  data.table::fwrite(
    df_clean,
    sprintf("xgb_BO_results_batch_%02d.csv", batch_id)
  )
  
  # Kompletter State als RDS sichern
  saveRDS(
    instance,
    sprintf("xgb_BO_batch_%02d.rds", batch_id)
  )
  
  # Beste bisherige Parameter ausgeben
  best <- instance$archive$best()
  message(sprintf(
    "✅ Beste Acc=%.4f | nrounds=%d | eta=%.3f | max_depth=%d | min_child_weight=%d | subsample=%.2f",
    best$classif.acc, best$nrounds, best$eta, best$max_depth,
    best$min_child_weight, best$subsample
  ))
}

message("🚀 Alle 50 Batches abgeschlossen – Loop beendet.")

library(data.table)

# 1.1) Liste aller Batch‑RDS‑Dateien
rds_files <- list.files(
  pattern   = "^xgb_BO_batch_.*\\.rds$",
  full.names = TRUE
)
print(rds_files)

# 1.2) Lade jede Instanz, ziehe archive$data heraus und säubere List‑Spalten
all_results <- rbindlist(lapply(rds_files, function(f) {
  inst <- readRDS(f)
  dt   <- inst$archive$data
  # entferne die List‑Spalten
  dt[, c("learner_param_vals","x_domain") := NULL]
  # gib Batch‑ID hinzu
  dt[, batch := sub(".*_batch_(\\d+)\\.rds$", "\\1", basename(f))]
  dt
}), fill = TRUE)

# 1.3) Speichere alles in einer CSV zur einfachen Übersicht
fwrite(all_results, "all_bo_results.csv")
message("▶ Alle bisherigen Ergebnisse in all_bo_results.csv gespeichert")

# 2.1) Lade die zusammengefassten Ergebnisse
res_all <- fread("all_bo_results.csv")

# 2.2) Spalte mit Accuracy finden (falls nicht genau 'classif.acc')
acc_col <- grep("acc", names(res_all), value = TRUE, ignore.case = TRUE)[1]
cat("▶ Verwende Accuracy-Spalte:", acc_col, "\n")

# 2.3) Besten Eintrag ermitteln
best_row <- res_all[which.max(get(acc_col))]
print(best_row)

# 2.4) Parameter und nrounds extrahieren
best_params <- list(
  eta              = best_row$eta,
  max_depth        = as.integer(best_row$max_depth),
  min_child_weight = as.integer(best_row$min_child_weight),
  subsample        = best_row$subsample,
  objective        = "multi:softprob",
  num_class        = length(label_levels),
  eval_metric      = "merror"
)
best_nrounds <- as.integer(best_row$nrounds)


library(xgboost)

# 3.1) Trainiere auf dem gesamten Trainingsset
dtrain <- xgb.DMatrix(
  data  = as.matrix(x_train[, best_set]),
  label = y_train
)
final_model <- xgb.train(
  params  = best_params,
  data    = dtrain,
  nrounds = best_nrounds,
  verbose = 1
)

# 3.2) Falls Du ein Test‑Label hast, messe Accuracy:
if (exists("y_test")) {
  dtest <- xgb.DMatrix(as.matrix(x_test[, best_set]))
  pred <- predict(final_model, dtest)
  mat  <- matrix(pred, ncol = length(label_levels), byrow = TRUE)
  idx  <- max.col(mat) - 1
  pred_lbl <- factor(label_levels[idx+1], levels = label_levels)
  cat("▶ Final Model Accuracy:", mean(pred_lbl == y_test), "\n")
}

# 3.3) Für Submission (ohne y_test)
if (exists("FAS_ids") && exists("x_test")) {
  dtest <- xgb.DMatrix(as.matrix(x_test[, best_set]))
  pred <- predict(final_model, dtest)
  mat  <- matrix(pred, ncol = length(label_levels), byrow = TRUE)
  idx  <- max.col(mat) - 1
  pred_lbl <- factor(label_levels[idx+1], levels = label_levels)
  submission <- data.table(id = FAS_ids, status_group = as.character(pred_lbl))
  fwrite(submission, "submission.csv")
  cat("▶ Submission.csv geschrieben\n")
}












## ===== Unbegrenzte neue Iterationen, manueller Abbruch ===================
library(mlr3); library(mlr3tuning); library(mlr3mbo)
library(bbotk); library(data.table)

# 0) Letzte Instanz laden (oder vorhandene 'instance' verwenden)
rds_files <- list.files(pattern = "^xgb_BO_batch_\\d+\\.rds$", full.names = TRUE)
if (length(rds_files)) {
  latest <- rds_files[order(rds_files, decreasing = TRUE)][1]
  message("🔄 Lade letzte Instanz: ", latest)
  instance <- readRDS(latest)
} else {
  stopifnot(exists("instance"))
}

stopifnot(exists("tuner"))  # sollte tnr("mbo") sein
dir.create("bo_runs", showWarnings = FALSE)

# 1) Terminierung praktisch "aus": sehr großes Evals-Limit
#    (Alternative, falls verfügbar: instance$terminator <- trm("none"))
bigN <- 1e9L
new_term <- trm("evals", n_evals = nrow(instance$archive$data) + bigN)
instance$terminator <- new_term

# 2) Batch-Zähler fortführen
existing_batches <- as.integer(gsub("^.*_batch_(\\d+)\\.rds$", "\\1", basename(rds_files)))
batch_id <- if (length(existing_batches)) max(existing_batches, na.rm = TRUE) else 0L

message("▶ Starte offenen BO-Loop. Beenden per ESC / Strg+C.")

repeat {
  batch_id <- batch_id + 1L
  message(sprintf("🔁 Starte Batch %02d – %s", batch_id, format(Sys.time(), "%H:%M:%S")))
  
  # genau 1 neue MBO-Konfiguration holen+bewerten (wie in deinem Setup)
  tuner$optimize(instance)
  
  # nur atomare Spalten für CSV
  df_clean <- copy(instance$archive$data)
  for (col in intersect(c("learner_param_vals","x_domain"), names(df_clean))) {
    df_clean[[col]] <- NULL
  }
  df_clean[, batch_nr := batch_id]
  
  # persistieren
  fwrite(df_clean, sprintf("bo_runs/xgb_BO_results_batch_%02d.csv", batch_id))
  saveRDS(instance, sprintf("xgb_BO_batch_%02d.rds", batch_id))
  
  # Fortschritt loggen
  best <- instance$archive$best()
  message(sprintf(
    "✅ Beste Acc=%.4f | nrounds=%d | eta=%.5f | max_depth=%d | min_child_weight=%d | subsample=%.3f",
    best$classif.acc, best$nrounds, best$eta, best$max_depth, best$min_child_weight, best$subsample
  ))
  
}





library(data.table)
library(xgboost)

## ---------- 1) RDS-Schnappschüsse finden (Batch 2..10) ----------
rds_files <- list.files(pattern = "^xgb_BO_batch_\\d+\\.rds$", full.names = TRUE)
stopifnot("Keine RDS-Batches gefunden." = length(rds_files) > 0)

# Batchnummer extrahieren
get_batch_num <- function(path) as.integer(sub(".*_batch_(\\d+)\\.rds$", "\\1", basename(path)))
rds_dt <- data.table(file = rds_files, batch = sapply(rds_files, get_batch_num))

# nur Batches 2..10 berücksichtigen, die wirklich existieren
rds_dt <- rds_dt[batch %in% 2:10][order(batch)]
stopifnot("Keine RDS-Batches im Bereich 2..10 gefunden." = nrow(rds_dt) > 0)


prev_needed <- unique(pmax(rds_dt$batch - 1L, 1L))
rds_prev <- data.table(file = list.files(pattern="^xgb_BO_batch_\\d+\\.rds$", full.names = TRUE))
rds_prev[, batch := sapply(file, get_batch_num)]
needed_batches <- sort(unique(c(rds_dt$batch, prev_needed)))
rds_needed <- rds_prev[batch %in% needed_batches][order(batch)]

# Hilfsfunktion: Archiv laden + aufräumen
load_archive <- function(f) {
  inst <- readRDS(f)
  dt   <- as.data.table(inst$archive$data)
  # störende List-Spalten entfernen
  for (col in intersect(c("learner_param_vals","x_domain"), names(dt))) dt[[col]] <- NULL
  dt
}

## ---------- 2) Pro Batch die "neue" Zeile (Delta) extrahieren ----------
# Wir vergleichen uhash zwischen Batch k und Batch (k-1) und nehmen die neue(n) uhash in k.
# Falls 'uhash' fehlt, erzeugen wir eine Notfall-ID aus den Parametern.
make_fallback_id <- function(dt) {
  keys <- intersect(c("nrounds","eta","max_depth","min_child_weight","subsample"), names(dt))
  if (length(keys) == 0L) return(NULL)
  apply(dt[, ..keys], 1, function(x) paste0(keys, "=", as.character(x), collapse="|"))
}

delta_rows <- list()

# Map: batch -> arch_dt
arch_map <- list()
for (i in seq_len(nrow(rds_needed))) {
  b  <- rds_needed$batch[i]
  f  <- rds_needed$file[i]
  arch_map[[as.character(b)]] <- load_archive(f)
}

for (b in rds_dt$batch) {
  dt_k  <- copy(arch_map[[as.character(b)]])
  dt_km <- arch_map[[as.character(b-1)]]
  # IDs
  if ("uhash" %in% names(dt_k) && "uhash" %in% names(dt_km)) {
    new_ids <- setdiff(dt_k$uhash, dt_km$uhash)
    row_k   <- dt_k[uhash %in% new_ids]
  } else {
    id_k  <- make_fallback_id(dt_k)
    id_km <- make_fallback_id(dt_km)
    stopifnot("Weder 'uhash' noch param-Spalten für Fallback-ID vorhanden." = !is.null(id_k) && !is.null(id_km))
    new_ids <- setdiff(id_k, id_km)
    row_k   <- dt_k[match(new_ids, id_k), ]
  }
  # Es sollte genau 1 neue Konfiguration sein (bei sequentieller Optimierung)
  row_k[, batch := b]
  delta_rows[[length(delta_rows) + 1L]] <- row_k
}

res_210 <- rbindlist(delta_rows, fill = TRUE)

## ---------- 3) Bestes Ergebnis (nur Batches 2..10) bestimmen ----------
need <- c("classif.acc","nrounds","eta","max_depth","min_child_weight","subsample","batch")
miss <- setdiff(need, names(res_210))
stopifnot(sprintf("Fehlende Spalten in Delta-Ergebnissen: %s", paste(miss, collapse=", ")) = length(miss) == 0)
stopifnot("Keine Delta-Einträge in Batches 2..10 gefunden." = nrow(res_210) > 0)

best_210 <- res_210[which.max(classif.acc)]

cat(sprintf(
  "🏆 Best (nur neu evaluierte Konfigurationen aus Batches 2–10)\n  acc=%.6f | batch=%d | nrounds=%d | eta=%.6f | max_depth=%d | min_child_weight=%d | subsample=%.6f\n",
  best_210$classif.acc, best_210$batch, as.integer(best_210$nrounds),
  best_210$eta, as.integer(best_210$max_depth), as.integer(best_210$min_child_weight),
  best_210$subsample
))
print(best_210[, .(batch, classif.acc, nrounds, eta, max_depth, min_child_weight, subsample)])

## ---------- 4) Finales Modell trainieren & vorhersagen ----------
stopifnot(exists("x_train"), exists("y_train"), exists("x_test"), exists("best_set"))

if (!exists("label_levels")) {
  if (is.factor(y_train)) {
    label_levels <- levels(y_train)
  } else stop("label_levels fehlt und y_train ist kein Factor.")
}

params <- list(
  eta              = as.numeric(best_210$eta),
  max_depth        = as.integer(best_210$max_depth),
  min_child_weight = as.integer(best_210$min_child_weight),
  subsample        = as.numeric(best_210$subsample),
  objective        = "multi:softprob",
  num_class        = length(label_levels),
  eval_metric      = "merror"
)
nrounds <- as.integer(best_210$nrounds)

dtrain <- xgb.DMatrix(
  data  = as.matrix(x_train[, best_set, drop = FALSE]),
  label = y_train
)
final_model <- xgb.train(params = params, data = dtrain, nrounds = nrounds, verbose = 1)

dtest <- xgb.DMatrix(as.matrix(x_test[, best_set, drop = FALSE]))
pred  <- predict(final_model, dtest)
mat   <- matrix(pred, ncol = length(label_levels), byrow = TRUE)
idx   <- max.col(mat) - 1
pred_lbl <- factor(label_levels[idx + 1], levels = label_levels)

# Optional: Metrik & Submission
if (exists("y_test")) {
  cat(sprintf("✅ Final-Accuracy (Holdout): %.4f\n", mean(pred_lbl == y_test)))
}
if (exists("FAS_ids")) {
  out_name <- sprintf("submission_batch_best_2to10.csv")
  fwrite(data.table(id = FAS_ids, status_group = as.character(pred_lbl)), out_name)
  cat(sprintf("💾 Geschrieben: %s\n", out_name))
}
