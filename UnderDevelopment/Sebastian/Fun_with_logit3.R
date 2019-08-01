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
fold1 <- read.table("/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/fold1_subset.csv",
                   sep = ",",
                   header = TRUE)
fold1 <- fold1[,!(names(fold1) %in% c('X','id'))] #Remove stupid ID
fold1 <- fold1[1:floor(nrow(fold1) / 10),] #Kun subset først 10.del for ikke at overfitte for sindssygt..

fold2 <- read.table("/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/fold2_subset.csv",
                    sep = ",",
                    header = TRUE)
fold2 <- fold2[,!(names(fold2) %in% c('X','id'))] #Remove stupid ID
fold2 <- fold2[1:floor(nrow(fold2) / 10),] #Kun subset først 10.del for ikke at overfitte for sindsygt..

## This has fewer columns
fold3 <- read.table("/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/fold3_subset.csv",
                    sep = ",",
                    header = TRUE)
fold3 <- fold3[,!(names(fold3) %in% c('X','id'))] #Remove stupid ID
fold3 <- fold3[1:floor(nrow(fold3) / 10),] #Kun subset først 10.del for ikke at overfitte for sindsygt..

data <- rbind.fill(fold1,fold2,fold3)

rm(list=c("fold1","fold2","fold3"))

### Start by scaling data ------------------------------------------------------------
data_scaled <- data
#Old scales
tmp <- c("TOTUSJH_last", "TOTPOT_last", "SAVNCPP_last", "XR_MAX_last")
data_scaled[tmp] <- as.data.frame(sign(data[tmp]) * abs(data[tmp])^(1/5))
tmp <- c("TOTFZ_last")
data_scaled[tmp] <- as.data.frame(sign(data[tmp]) * abs(data[tmp])^(1/2))
#New scales
tmp <- c("TOTUSJH_dx", "TOTFZ_dx", "SAVNCPP_dx")
data_scaled[tmp] <- as.data.frame(sign(data[tmp]) * abs(data[tmp])^(1/2))
tmp <- c("R_VALUE_dx", "XR_MAX_dx", "TOTPOT_dx")
data_scaled[tmp] <- as.data.frame(sign(data[tmp]) * abs(data[tmp])^(1/3))

tmp <- lapply(names(data_scaled), function(x) 
  ggplot(data = data_scaled[, x, drop = FALSE]) + 
    aes_string(x) + xlab(x) + ylab(""))
gd <- geom_density(adjust = 2, fill = gray(0.5))
gh <- function(width) {geom_histogram(binwidth = width)}
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
  nrow = 2
)

grid.arrange(
  tmp[[67]] + gd,
  tmp[[68]] + gd,
  tmp[[69]] + gd,
  tmp[[70]] + gd,
  tmp[[71]] + gd,
  tmp[[72]] + gd,
  tmp[[73]] + gd,
  tmp[[74]] + gd,
  tmp[[75]] + gd,
  nrow = 2
)

### FORWARD BIC ----------------------------------------------------------------------

full_form <- label ~ TOTUSJH_last*TOTUSJH_dx + 
  XR_MAX_last*XR_MAX_dx*NA_XR_MAX_last + 
  R_VALUE_last*R_VALUE_dx*NA_Rmask_last + 
  TOTPOT_last*TOTPOT_dx + 
  MEANGBT_last*MEANGBT_dx + 
  MEANGBZ_last*MEANGBZ_dx + 
  MEANGAM_last*MEANGAM_dx + 
  TOTFZ_last*TOTFZ_dx + 
  SAVNCPP_last*SAVNCPP_dx
nulModel <- glm(label ~ 1, data = data_scaled, family = binomial(link = "logit"))
fullModel <- glm(full_form, data = data_scaled, family = binomial(link = "logit"))
summary(fullModel)

n_samples <- nrow(data_scaled)
forward_BIC <- step(nulModel, 
                    direction = "forward", 
                    k = log(n_samples), #BIC
                    scope = list(upper = fullModel),
                    trace = 0 #How much information to print
)
summary(forward_BIC)
forward_back <- step(forward_BIC, 
                     direction = "backward", 
                     k = log(n_samples), #BIC
                     trace = 0 #How much information to print
)
summary(forward_back)

##### Create final model
final_form <- forward_back$formula
final_model <- glm(final_form, data = data_scaled, family = binomial(link = "logit"))
##label ~ TOTUSJH_last + R_VALUE_last + XR_MAX_last + MEANGBZ_last + 
#SAVNCPP_last + TOTFZ_last + TOTPOT_last + MEANGBT_last + 
#  MEANGAM_last + SAVNCPP_dx + NA_Rmask_last + NA_XR_MAX_last


#### TJEK LINEARITET --------------------------------------------------------------------------------
modelDiag <- transform(data_scaled,
                     .fitted = predict(forward_back),
                     .deviance = residuals(forward_back))

grid.arrange(
  qplot(.fitted, .deviance, data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(TOTUSJH_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(XR_MAX_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(R_VALUE_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(TOTPOT_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(TOTFZ_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(SAVNCPP_dx, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(MEANGAM_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(SAVNCPP_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(MEANGBZ_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(MEANGBT_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  nrow = 3
) 
# Det ser egentligt ret fint ud. Beholder denne som sidste model. 

#### Test_set
test_set <- read.table("/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/testSet_subset.csv",
                   sep = ",",
                   header = TRUE)
test_set <- test_set[,!(names(test_set) %in% c('X','id'))] #Remove stupid ID

#Scale:
test_set_scaled <- test_set

#Old scales
tmp <- c("TOTUSJH_last", "TOTPOT_last", "SAVNCPP_last", "XR_MAX_last")
test_set_scaled[tmp] <- as.data.frame(sign(test_set[tmp]) * abs(test_set[tmp])^(1/5))
tmp <- c("TOTFZ_last")
test_set_scaled[tmp] <- as.data.frame(sign(test_set[tmp]) * abs(test_set[tmp])^(1/2))
#New scales
tmp <- c("TOTUSJH_dx", "TOTFZ_dx", "SAVNCPP_dx")
test_set_scaled[tmp] <- as.data.frame(sign(test_set[tmp]) * abs(test_set[tmp])^(1/2))
tmp <- c("R_VALUE_dx", "XR_MAX_dx", "TOTPOT_dx")
test_set_scaled[tmp] <- as.data.frame(sign(test_set[tmp]) * abs(test_set[tmp])^(1/3))

## Predict
predictions <- predict(final_model, newdata = test_set_scaled, type = 'response')
binary_preds <- numeric(length(predictions))
binary_preds[predictions > 0.35] <- 1

### Export to csv
#ones <- rep(1,length(predictions))
ones <- data.frame(rep(1,length(predictions)))
colnames(ones) <- c("ClassLabel")

write.csv(ones, 
          file = "/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/ones.csv",
          row.names = TRUE,
          quote = FALSE)

preds_frame <- data.frame(binary_preds)
colnames(preds_frame) <- c("ClassLabel")
write.csv(preds_frame, 
          file = "/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/logit_predictions0.csv",
          row.names = TRUE,
          quote = FALSE)
