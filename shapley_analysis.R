# this script produces charts on variable importance 
source("helpers/utils.R")
exp_name <- "test_case"

df <- read.csv(paste0("results/aggregated/shapley_forecast_", exp_name, ".csv"))
df$date <- as.Date(df$date)
df <- df %>% filter(date >= periods[["all"]][1] & date <= periods[["all"]][2])

# get variable names
shap_vars <- grep("shap_", colnames(df), value = T)
vars <- gsub("shap_", "", shap_vars)

#### Plotting Shapley Shares (Global importance measure) ####
unique_exps <- unique(df$method)
mean_shap <- t(sapply(unique_exps, function(x) shap_importance(df[df$method == x, shap_vars])))
n_features <- length(shap_vars)
methods_show <- c(
  "LightGBM",
  "Ridge"
  )



margin <- c(8.5, 3, 2, .1)
# order by average 
oo <- order(apply(mean_shap, 2, mean), decreasing = T) # order of variables
par(mar = margin)
plot.new()
plot.window(xlim = c(1, length(shap_vars)), 
            ylim = c(-.075, .33))
abline(h = 0, col = "gray50")
axis(1, at = 1:length(shap_vars), labels = F)
text(x = 1:length(shap_vars), y = par("usr")[3] - 0.04, labels = feature_names[vars][oo], xpd = NA, srt = 65, cex = 1, adj = 1)
axis(2)
title(main = "Shapley share", line = 0)
ymin <- rep(1, length(shap_vars))
ymax <- rep(0, length(shap_vars))
for(method in methods_show){
  ix_row <- method
  ymax <- pmax(ymax, mean_shap[ix_row, ])
  ymin <- pmin(ymin, mean_shap[ix_row, ])
}
segments(x0 = 1:length(shap_vars), y0 = ymin[oo], y1 = ymax[oo], col = "gray50")
for(method in methods_show){
  ix_row <- method
  lines(mean_shap[ix_row, oo], pch = pch_algos[method], col = cols_algos[method], type = ifelse(method == methods_show[1] ,"o", "p"), lwd = 1.2)
  
}
legend("topright", legend = clean_names[methods_show], 
       pch = pch_algos[methods_show],
       col = cols_algos[methods_show],
       lty = 1,
       bty = "n",
       y.intersp = .8,
       cex = .95
)


#### Plotting Permutation Importance ###
permutation_name <- c("absolute_error" = "Mean permutation\nvalues (Absolute error)",
                      "squared_error" = "Mean permutation\nvalues (Squared error)",
                      "prediction deviance" = "Mean permutation\nvalues (Deviance)")

dfp <- read.csv(paste0("results/aggregated/permutation_forecast_", exp_name, ".csv"))

permutation_type <- "prediction deviance" # "absolute_error"
par(mar = margin)
plot.new()
plot.window(xlim = c(1, length(shap_vars)), 
            ylim = c(-.075, .42))
abline(h = 0, col = "gray50")
axis(1, at = 1:length(shap_vars), labels = F)
text(x = 1:length(shap_vars), y = par("usr")[3] - 0.04, labels = feature_names[vars][oo], xpd = NA, srt = 65, cex = 1, adj = 1)
axis(2)
title(main = permutation_name[permutation_type], line = 0)

ymin <- rep(1, length(shap_vars))
ymax <- rep(0, length(shap_vars))

vals_list <- list()
for(method in methods_show){
  ix_row <- dfp$method == method & 
    dfp$type == permutation_type
  vals <- dfp[ix_row, vars]
  
  if (permutation_type %in% c("absolute_error", "squared_error"))
    vals <- vals - 1
  
  vals <- vals/ sum(vals)
  vals_list[[method]] <- vals
  
  ymax <- pmax(ymax, vals)
  ymin <- pmin(ymin, vals)
}
par(xpd = T)
segments(x0 = 1:length(vars), y0 = ymin[oo], y1 = ymax[oo], col = "gray50")
for(method in methods_show)
  lines(as.matrix(vals_list[[method]][1, oo])[1,], pch = pch_algos[method], col = cols_algos[method], type = ifelse(method == methods_show[1], "o", "p"))


#### Shapley scatter plots ####
# this is based on bootstrapped analysis #

df <- read.csv(paste0("results/aggregated/shapley_out_of_bag_", exp_name, ".csv"))
df$date <- as.Date(df$date)
df <- df[df$period == "2019-11-01",]

df <- df %>% filter(date >= periods[["all"]][1] & date <= periods[["all"]][2])


methods_shap_scatter <- c("LightGBM", "Ridge")
vars_shap_scatter <- c(vars[oo][1:4])
par(mar = c(3,4,1,.5), mfrow = c(length(vars_shap_scatter), length(methods_shap_scatter)))
for (v in vars_shap_scatter) {
  for(m in methods_shap_scatter){
    ix <- df$method == m
    dd <- df[ix,]
    shap_scatter_single(dd, v = v, polynomial_degree = 3, col = cols_algos[m], 
                        ylim = c(-1, 1),
                        xlab = ifelse(v == vars_shap_scatter[length(vars_shap_scatter)], "Observed values", ""),
                        ylab = ifelse(m == methods_shap_scatter[1], feature_names[v], ""),
                        main = ifelse(v == vars_shap_scatter[1], clean_names[m], "")
    )
  }
}
