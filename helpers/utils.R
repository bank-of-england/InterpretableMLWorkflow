library(tidyverse)



figure_path <- "C:/Users/marcu/Dropbox/boe_research/macroforecasting/paper&slides/IJCB_RR/figures/"



periods = list(
  "all" = c("1990-01-01", "2019-11-01"),
  "01/1990 - 12/1999" = c("1990-01-01", "1999-12-01"),
  "01/2000 - 08/2008"= c("2000-01-01", "2008-08-01"),
  "09/2008 - 11/2019"= c("2008-09-01", "2019-11-01")
)




cols_periods <- c("01/1990 - 12/1999" = "blue",
                  "01/2000 - 08/2008"= "red",
                  "09/2008 - 11/2019"= "darkgoldenrod4")


clean_names <- c("Forest" = "Random forest",
                 "NN" = "Neural network",
                 "NN_refined" = "Neural network",
                 "OLS" = "OLS regression",
                 "Ridge" = "Ridge regression",
                 "Lasso" = "Lasso regression",
                 "SVM" = "SVR",
                 "AR1" = "AR1",
                 "AR_auto" = "AR12",
                 "LightGBM" = "Gradient boosting",
                 "LightGBM_dart" = "Gradient boosting",
                 "LightGBM_goss" = "Gradient boosting"
)




cols_algos <- c("true" = "black", 
                "OLS" = "gray40", 
                "Lasso" = "gray40", 
                "Ridge" = "gray40", 
                "AR1" = "chartreuse4", 
                "AR_auto" = "chartreuse4", 
                "AR" = "chartreuse4",
                "NN" = "coral4",
                "Forest" = "red",
                "LightGBM" = "blue",
                "SVM" = "darkorange2"
)


pch_algos <- c( # "true" = "black", 
  "OLS" = 9, 
  "Lasso" = 8, 
  "Ridge" = 0, 
  "AR1" = 6, 
  "AR_auto" = 5, 
  "LightGBM" = 4,
  "NN" = 2,
  "Forest" = 3,
  "SVM" = 1
)



group_names <- c("consumption_orders" = "Consumption and orders",
                 "interest_exchange" = "Interest rate and exchange rates", 
                 "labor_market" = "Labour market",
                 "money_credit" = "Money and credit",
                 "orders_inventories" = "Orders and inventories",
                 "output_income" = "Output and income",
                 "prices" = "Prices",
                 "stock_market" = "Stock market")




split_name <- function(x){
  
  out <- gsub(" ", "\\\n", x)
  out[-grep("\\n", out)] <- paste0(out[-grep("\\n", out)], "\n") # add line break to single word entries as well
  return(out)
}



feature_names <- c( # for convenience: ordered by mean importance
  "INDPRO" = "Industrial production",
  "S.P.500" = "S&P 500",
  "DPCERA3M086SBEA" = "Consumption",
  "UNRATE" = "Unemployment rate",
  "BUSLOANS" = "Business loans",
  "TB3MS" = "3-month trasury bill",
  "RPI" = "Real personal income",
  "OILPRICEx" =  "Oil price",
  "M2SL" = "M2 Money",
  "CPIAUCSL" = "CPI",
  
  # additional features
  "USGOOD" = "Employees (goods-producing)",
  "PAYEMS" = "Employees (total)",
  "MANEMP" = "Employees (manufacturing)",
  "DMANEMP"= "Employees (durable goods)",
  "IPMANSICS" = "Industrial production (manufacturing)",
  "T10YFFM" = "10-year treasury rate\nminus federal funds rate",
  "T5YFFM" = "5-year treasury rate\nminus federal funds rate",
  "AAAFFM" = "Moody's Aaa Corporate Bonds - federal funds rate",
  "BAAFFM" = "Moody's Baa Corporate Bonds - federal funds rate",
  "TB6SMFFM" = "6-month Treasury C - federal funds rate",
  "CPITRNSL" = "CPI (transport)",
  "IPBUSEQ" = "Industrial Production\n(Business equipment)"
)




# highlight best model in a table
highlight_best <- function(x, digits = 2){
  x <- data.frame(x) # does not work with tibbles
  x_out <- x
  for(i in 1:nrow(x)){
    x_out[i,] <- ifelse(round(x[i,],2) == min(round(x[i,], digits), na.rm = T), 
                        paste0("\\textbf{", format(x[i,],nsmall = digits, digits = digits), "}"),
                        format(x[i,],nsmall = digits, digits = digits)
    )
  }
  return(x_out)
}



# compute prediction error given the observed (y) and predicted (pred) response
compute_metrics <- function(y, pred){
  output = c(
    "abs_error" = mean(abs(y - pred)),
    "abs_error_sum" = sum(abs(y - pred)),
    "rmse" = sqrt(sum((y - pred)^2) / length(y)),
    "cor" = cor(y, pred),
    "direction_acc" = mean(sign(y) == sign(pred)), 
    "direction_acc_no0" = mean(sign(y[y!=0]) == sign(pred[y!=0])), #  # not clear how to treat non-changes... see: https://en.wikipedia.org/wiki/Mean_directional_accuracy
    "n" = length(y)
  )
  return(output)         
}


# Diebold Mariano test comparing whether the error of the prediction of one method (pred1)
# is significantly different form the error of the prediction of the other method (pred2)
dm_test <- function(true, pred1, pred2, alternative = "greater"){
  if(all(pred1 == pred2)){
    return(1)
  } else {
    out <- forecast::dm.test(ts(true-pred1), ts(true-pred2), alternative = alternative, h= 1, power = 1)
    # print(out)
    return(unname(out$p.value))
  }
}


# plot the time series of the predcition error of different methods
plot_time_series_error <- function(data, methods = c("Forest", "OLS", "AR1"), absolute_error = F){
  if(is.null(data$counter))
    data$counter <- 0
  counters <- unique(data$counter)
  plot.new()
  plot.window(xlim = c(min(data$date), max(data$date)),
              ylim = c(ifelse(absolute_error, 0, -4), 3))
  xdates <- as.Date(paste0(1990:2019, "-01-01"))
  axis(1, at = xdates, labels = F)
  text(x = xdates,
       y = par("usr")[3] - 0.75,
       labels = 1990:2019,
       xpd = NA,
       srt = 45,
       cex = .8)
  title(ylab = "Error (observed - predicted)", line = 2)
  
  axis(2)
  abline(v = as.Date(c("2008-09-01","2000-01-01")), col = "gray50")
  abline(h = 0, col = "gray50")
  
  dd <- data[data$method == methods[1] & data$counter == counters[1],]
  
    for(method in methods){
      for(d in counters){
        dd <- data[data$method == method & data$counter == d,]
        y <- dd$pred - dd$true
        if(absolute_error)
          y <- abs(y)
        lines(dd$date,
              y, 
              col = makeTransparent(cols_algos[method], alpha = .82)
        )
      }
  
  }
}

# plots the time series of the 
plot_time_series <- function(data, methods = c("Forest", "OLS", "AR1"), show_target = T){
  if(is.null(data$counter))
    data$counter <- 0
  counters <- unique(data$counter)
  plot.new()
  plot.window(xlim = c(min(data$date), max(data$date)),
              ylim = c(-2, 4))
  xdates <- as.Date(paste0(1990:2019, "-01-01"))
  axis(1, at = xdates, labels = F)
  text(x = xdates,
       y = par("usr")[3] - 0.75,
       labels = 1990:2019,
       xpd = NA,
       srt = 45,
       cex = .8)
  title(ylab = "1-year change in unemployment", line = 2)
  
  axis(2)
  abline(v = as.Date(c("2008-09-01","2000-01-01")), col = "gray50")
  abline(h = 0, col = "gray50")
  if(show_target){
    legend("topright", lty = 1, legend = c("Observed", clean_names[methods]), col = c("black", cols_algos[methods]), lwd = c(2, rep(1, length(methods))), bty = "n")
  } else {
    legend("topright", lty = 1, legend = clean_names[methods], col = cols_algos[methods], bty = "n")
  }
  
  dd <- data[data$method == methods[1] & data$counter == counters[1],]
  if(show_target)
    lines(dd$date, dd$true, col = cols_algos["true"], lwd = 2)
  
    for(method in methods){
      for(d in counters){
        dd <- data[data$method == method & data$counter == d,]
        lines(dd$date,
              dd$pred, 
              col = makeTransparent(cols_algos[method], alpha = .82)
        )
      }
    }
  
  if(length(counters) > 1 & length(methods) == 1){
    dd <- data[data$method == methods, ]
    dd <- dd %>% group_by(date) %>% summarise(min = min(pred), max = max(pred))
    segments(x0 = dd$date, y0 = dd$min, y1 = dd$max, lwd = 2, col = makeTransparent(cols_algos[method], alpha = .5))
  }
}

makeTransparent <- function(..., alpha=0.5) {
  if(alpha<0 | alpha>1) stop("alpha must be between 0 and 1")
  alpha = floor(255*alpha)
  newColor = col2rgb(col=unlist(list(...)), alpha=FALSE)
  .makeTransparent = function(col, alpha) {
    rgb(red=col[1], green=col[2], blue=col[3], alpha=alpha, maxColorValue=255)
  }
  newColor = apply(newColor, 2, .makeTransparent, alpha=alpha)
  return(newColor)
}


shap_importance <- function(shap_matrix){
  row_sum <- apply(abs(shap_matrix), 1, sum, na.rm = T)
  shap_mean_norm <- shap_matrix / replicate(ncol(shap_matrix), row_sum)
  return(apply(abs(shap_mean_norm), 2, mean, na.rm = T))
}



shap_scatter_single <- function(data_input, v, 
                                ylim = NULL,
                                polynomial_degree = 0,
                                ylab = "",
                                xlab = "Observed predictor values",
                                main = feature_names[v],
                                main_line = 0,
                                col = "gray50",
                                col_function = "black",
                                sizer = 1,
                                add = FALSE,
                                grid = T# add to plot without creating new plot
                                
){
  if(!add){
    xlim <- c(min(data_input[[v]]), max(data_input[[v]]))
    if(is.null(ylim[1])){
      ylim_use <-  c(min(data_input[[paste0("shap_", v)]]), max(data_input[[paste0("shap_", v)]]))
    } else {
      ylim_use <- ylim
    }
    
    plot.new()
    
    plot.window(xlim = xlim, 
                ylim = ylim_use)
    axis(1, cex.axis = sizer); axis(2, cex.axis = sizer)
    
    if(ylab != ""){
      title(ylab = "Shapley values", line = 2, cex.lab = sizer)
      title(ylab = ylab, line = 3, cex.lab = sizer, font.lab = 2)
    }
    title(xlab = xlab, line = 2, cex.lab = sizer)
    title(main = main, line = main_line)
    if(grid){
      abline(h = 0, col = "gray50", lwd = .5)
      abline(v = 0, col = "gray50", lwd = .5)
    }
    
  }
  points(data_input[[v]], data_input[[paste0("shap_", v)]], pch = 20, col = makeTransparent(col, alpha = .5))
  if(polynomial_degree > 0){
    poly_fit <- lm(data_input[[paste0("shap_", v)]]~poly(data_input[[v]], degree = polynomial_degree))
    poly_fit$fitted.values
    oo <- order(data_input[[v]])
    lines(data_input[[v]][oo], poly_fit$fitted.values[oo], col = col_function)
  }
  
}

shap_scatter_methods <- function(data_input, v, methods, ylim = c(-1, 1.2), show_series = c(), polynomial_degree = 0){
  xlim <- c(min(data_input[[v]]), max(data_input[[v]]))
  plot.new()
  plot.window(xlim = xlim, ylim = ylim)
  axis(1); axis(2)
  title(ylab = "Shapley value", xlab = "Observed predictor values", line = 2)
  title(main = feature_names[v], line = 0)
  for (method in methods){
    ix <- data_input$method == method
    print(sum(ix))
    if(method %in% show_series)
      points(data_input[[v]][ix], data_input[[paste0("shap_", v)]][ix], 
             pch = pch_algos[method],
             col = makeTransparent(cols_algos[method], alpha = .4),
             cex = .7
      )  
    
    if(polynomial_degree > 0){
      poly_fit <- lm(data_input[[paste0("shap_", v)]][ix] ~ poly(data_input[[v]][ix], polynomial_degree))
      oo <- order(data_input[[v]][ix])
      lines(data_input[[v]][ix][oo], poly_fit$fitted.values[oo], col = cols_algos[method])
    }
  }
  legend("bottomleft", legend = clean_names[methods], col = cols_algos[methods], bty = "n", y.intersp = .8, lty = 1)
}

