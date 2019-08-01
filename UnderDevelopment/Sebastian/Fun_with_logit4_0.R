### THIS CONTINUES FROM 'Fun_with_logit2'.
## Here the real predictions on the testset will be made
## They are based on equal amount from fold1, fold2 and fold3.
## The threshold is set to 0.35 (based on earlier plots, between 0.3 and 0.4 doesn't reaaly make a difference)

rm(list=ls())
set.seed(1500)

library(ggplot2)   ## Grammar of graphics
library(reshape2)  ## Reshaping data frames
library(lattice)   ## More graphics
library(hexbin)    ## and more graphics
library(gridExtra) ## ... and more graphics
library(xtable)    ## LaTeX formatting of tables
library(splines)   ## Splines -- surprise :-)
library(survival)  ## Survival analysis
library(grid)      ## For 'unit'
library(lpSolve)   ## Linear programming
library(plyr)      ## to rbind dataframes with missing columns

### Korrelationsplot og hexabin-scale
cor.print <- function(x, y) {
  panel.text(mean(range(x)), mean(range(y)),
             paste('$', round(cor(x, y), digits = 2), '$', sep = '')
  )
}
binScale <- scale_fill_continuous(breaks = c(1, 10, 100, 1000),
                                  low = "gray80", high = "black",
                                  trans = "log", guide = "none")

### READ DATA ------------------------------------------------------------------------------------
fold1 <- read.table("/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/fold1_NA_features.csv",
                   sep = ",",
                   header = TRUE)
fold1 <- fold1[,!(names(fold1) %in% c('X','id'))] #Remove stupid ID
fold1 <- fold1[1:floor(nrow(fold1) / 10),] #Kun subset først 10.del for ikke at overfitte for sindssygt..
data <- fold1
colnames(data)

tmp <- lapply(names(data), function(x) 
  ggplot(data = data[, x, drop = FALSE]) + 
    aes_string(x) + xlab(x) + ylab(""))
gd <- geom_density(adjust = 2, fill = gray(0.5))
gh <- function(width) {geom_histogram(binwidth = width)}

#last values (fine)
grid.arrange(
  tmp[[98]] + gd,
  tmp[[99]] + gd,
  tmp[[100]] + gd,
  tmp[[101]] + gd,
  tmp[[102]] + gd,
  tmp[[103]] + gd,
  tmp[[104]] + gd,
  tmp[[105]] + gd,
  tmp[[106]] + gd,
  tmp[[107]] + gd,
  tmp[[112]] + gd,
  tmp[[113]] + gd,
  nrow = 4
)

#dx values (fine):
grid.arrange(
  tmp[[2]] + gd,
  tmp[[3]] + gd,
  tmp[[4]] + gd,
  tmp[[5]] + gd,
  tmp[[6]] + gd,
  tmp[[7]] + gd,
  tmp[[8]] + gd,
  tmp[[9]] + gd,
  tmp[[10]] + gd,
  tmp[[11]] + gd,
  tmp[[16]] + gd,
  tmp[[17]] + gd,
  nrow = 4
)

#dx^2 values (kan også godt lige accepteres, i hvert fald de første 3):
grid.arrange(
  tmp[[18]] + gd,
  tmp[[19]] + gd,
  tmp[[20]] + gd,
  tmp[[21]] + gd,
  tmp[[22]] + gd,
  tmp[[23]] + gd,
  tmp[[24]] + gd,
  tmp[[25]] + gd,
  tmp[[26]] + gd,
  tmp[[27]] + gd,
  tmp[[32]] + gd,
  tmp[[33]] + gd,
  nrow = 4
)

#dw values (fine efter log):
grid.arrange(
  tmp[[34]] + gd,
  tmp[[35]] + gd,
  tmp[[36]] + gd,
  tmp[[37]] + gd,
  tmp[[38]] + gd,
  tmp[[39]] + gd,
  tmp[[40]] + gd,
  tmp[[41]] + gd,
  tmp[[42]] + gd,
  tmp[[43]] + gd,
  tmp[[48]] + gd,
  tmp[[49]] + gd,
  nrow = 4
)

#dw2 values (lort)
grid.arrange(
  tmp[[50]] + gd,
  tmp[[51]] + gd,
  tmp[[52]] + gd,
  tmp[[53]] + gd,
  tmp[[54]] + gd,
  tmp[[55]] + gd,
  tmp[[56]] + gd,
  tmp[[57]] + gd,
  tmp[[58]] + gd,
  tmp[[59]] + gd,
  tmp[[64]] + gd,
  tmp[[65]] + gd,
  nrow = 4
)

temp <- data
my_cols <- c("pca_1_dw2","pca_2_dw2", "pca_3_dw2", "pca_4_dw2", "pca_5_dw2", "pca_6_dw2", "pca_7_dw2", "pca_8_dw2", "pca_9_dw2", "pca_10_dw2")
temp[,my_cols] <- sign(temp[,my_cols])* abs(temp[,my_cols])^(1/3)
tmp <- lapply(names(temp), function(x) 
  ggplot(data = temp[, x, drop = FALSE]) + 
    aes_string(x) + xlab(x) + ylab(""))
gd <- geom_density(adjust = 2, fill = gray(0.5))
gh <- function(width) {geom_histogram(binwidth = width)}

grid.arrange(
  tmp[[34]] + gd,
  tmp[[35]] + gd,
  nrow = 1
)

data_last <- data[,c(98,99,100,101,102,103,104,105,106,107,112,113)]
data_dx <- data[,c(2,3,4,5,6,7,8,9,10,11,16,17)]

#### Spearman plot fordi jeg ikke har normaliseret i dette her
data_sub <- data_dx[,!(names(data_last) %in% c('label', 'NA_SHARPmask_last', 'NA_Rmask_last', 'NA_satellite_last','NA_XR_MAX'))] #Remove stupid ID
cp <- cor(data.matrix(data_sub), method = "spearman")
ord <- rev(hclust(as.dist(1 - abs(cp)))$order)
colPal <- colorRampPalette(c("blue", "yellow"), space = "rgb")(100)
levelplot(cp[ord, ord],  
          xlab = "", 
          ylab = "",
          col.regions = colPal, 
          at = seq(-1, 1, length.out = 100),
          colorkey = list(space = "top", labels = list(cex = 1.5)),
          scales = list(x = list(rot = 45), 
                        y = list(rot = 0),
                        cex = 1.2)
)

### FORWARD BIC ----------------------------------------------------------------------
#### validate on fold2
fold2 <- read.table("/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/fold2_NA_features.csv",
                    sep = ",",
                    header = TRUE)
fold2 <- fold2[,!(names(fold2) %in% c('X','id'))] #Remove stupid ID
validation_set <- fold2

data_subset <- data
full_form <- label ~ pca_1_last + pca_2_last + pca_3_last + pca_4_last + pca_5_last +
  pca_6_last + pca_7_last + pca_8_last + pca_9_last + pca_10_last +
  pca_1_dx + pca_2_dx + pca_3_dx + pca_4_dx + pca_5_dx + pca_6_dx + pca_7_dx + pca_8_dx +
  pca_9_dx + pca_10_dx + pca_1_dx2 + log(pca_1_dw) +
  XR_MAX_last + R_VALUE_last + 
  NA_SHARPmask_last + NA_Rmask_last + NA_satellite_last + NA_XR_MAX_last

nulModel <- glm(label ~ 1, data = data_subset, family = binomial(link = "logit"))
fullModel <- glm(full_form, data = data_subset, family = binomial(link = "logit"))
summary(fullModel)

n_samples <- nrow(data_subset)
forward_BIC <- step(nulModel, 
                    direction = "forward", 
                    k = log(n_samples),
                    scope = list(upper = fullModel),
                    trace = 0 #How much information to print
)
summary(forward_BIC)
forward_back <- step(forward_BIC, 
                     direction = "backward", 
                     k = log(n_samples),
                     trace = 0 #How much information to print
)
summary(forward_back)

final_form <- forward_back$formula
final_model <- glm(final_form, data = data_subset, family = binomial(link = "logit"))


## Predict
predictions <- predict(final_model, newdata = validation_set, type = 'response')
correct <- fold2$label

plot_matrix <- f1_scores_plot(predictions,correct, 100)
max(plot_matrix[,2]) #Get maximum value
