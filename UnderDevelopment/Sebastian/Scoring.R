f1_scores_plot <- function(pred, true_vals,k) {
  
  pred_quant <- seq(max(pred)+0.001,min(pred)-0.001,(min(pred)-0.001-(max(pred)+0.001))/k)
  x_akse <- numeric(length = k+1)
  y_akse <- numeric(length = k+1)
  
  for (i in 1:k+1) {
    predictions <- data.frame(pred)
    predictions[pred_quant[i]>=pred,] <- 0 #Threshold større end prediction
    predictions[pred_quant[i]<pred,] <- 1 #Threshold mindre end prediction
    
    x_akse[i] <- pred_quant[i]
    
    TP <- length(predictions[predictions == 1 & true_vals == 1,])
    FP <- length(predictions[predictions == 1 & true_vals == 0,])
    FN <- length(predictions[predictions == 0 & true_vals == 1,])
    
    precision <- TP / (TP + FP)
    recall <- TP / (TP + FN)
    
    y_akse[i] <- 2 * precision * recall / (precision + recall)
    
  }
  plot(x_akse, y_akse)
  
  #x- og y-liste
  my_matrix <- matrix(nrow = length(x_akse), ncol = 2)
  my_matrix[,1] <- x_akse
  my_matrix[,2] <- y_akse
  
  return(my_matrix)
}
