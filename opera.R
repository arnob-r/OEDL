 library(opera)
 intermittency_combine_actual_vs_forecast <- read.delim("~/Desktop/intermittency_combine_actual_vs_forecast.txt", header=FALSE)
 View(intermittency_combine_actual_vs_forecast)
 Y <- intermittency_combine_actual_vs_forecast$V1
 X <- cbind(intermittency_combine_actual_vs_forecast$V2, intermittency_combine_actual_vs_forecast$V3, intermittency_combine_actual_vs_forecast$V4)
 matplot(cbind(Y, X), type = "l", col = 1:4, ylab = "Observable", xlab = "time", main = "Expert forecasts and observations")
 
 oracle.convex <- oracle(Y = Y, experts = X, loss.type = "square", model = "convex")
 print(oracle.convex)