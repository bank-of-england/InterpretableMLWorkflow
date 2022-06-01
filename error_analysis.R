# this script computes the prediction error and shows the time series
library(tidyverse)
source("helpers/utils.R")

exp_name <- "test_case" # name of the set of experiments for which we load the results
df_pred <- read.csv(paste0("results/aggregated/pred_all_", exp_name , "_raw.csv"))[,-1]
df_pred$date <- as.Date(df_pred$date)
df_pred <- df_pred %>% filter(date >= periods$all[1] & date <= periods$all[2])


# average results across repeated replications of the same experiment
df_pred_mean <- df_pred %>% group_by(date, method, hyper_type, features, lag,
                                     n_boot, winsorize, window_size) %>% summarise_at(c("pred", "true"), mean) 


# create a wider data set, where each column show the predictions of the different models
df_pred_wide <- df_pred_mean[, c("method", "hyper_type", "n_boot", "winsorize", "date", "true", "pred", "lag", "window_size")] %>% 
  pivot_wider(names_from = c(method, hyper_type, n_boot, winsorize, lag, window_size),
              values_from = "pred")

# compute prediction errors
exp_eval <- colnames(df_pred_wide)[- c(1:2)]
errors <- data.frame(t(sapply(exp_eval, function(x) compute_metrics(df_pred_wide$true, df_pred_wide[[x]]))))
methods_order <- rownames(errors)[order(errors$abs_error)]
errors[methods_order, ]

#### plot time series ####

par(mar = c(2.5,3,0,.5))

# shows predictions
plot_time_series(df_pred_mean, methods = c("LightGBM", "Ridge"), show_target = T)

# shows error
plot_time_series_error(df_pred_mean, methods = c("LightGBM", "Ridge", "AR1")); abline()


# statistical comparison of forecasting performance using the Diebold Mariano test

p_value <- dm_test(df_pred_wide$true, df_pred_wide$Ridge_kfold_block_gap_30_0.01_12_10000, df_pred_wide$LightGBM_kfold_block_gap_30_0.01_12_10000)





