### THIS CONTINUES FROM 'Fun_with_logit'.
### It is based on the variables from the 'final_form' and tries to see if the features should be expanded from last-values

#final_form <- label ~ TOTUSJH_last + XR_MAX_last*NA_XR_MAX_last + R_VALUE_last*NA_Rmask_last + 
#  TOTPOT_last + MEANGBT_last + MEANGBZ_last + MEANGAM_last + 
#  TOTFZ_last + SAVNCPP_last

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

### Korrelationsplot og hexabin-scale
cor.print <- function(x, y) {
  panel.text(mean(range(x)), mean(range(y)),
             paste('$', round(cor(x, y), digits = 2), '$', sep = '')
  )
}
binScale <- scale_fill_continuous(breaks = c(1, 10, 100, 1000),
                                  low = "gray80", high = "black",
                                  trans = "log", guide = "none")

### DATA
data <- read.table("/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/fold1_subset.csv",
                   sep = ",",
                   header = TRUE)
data <- data[,!(names(data) %in% c('X','id'))] #Remove stupid ID
data <- data[1:10000,] #Kun subset til at kigge på i startenª
colnames(data)

## We know TOTUSJH has the most influence and start by looking at the features from this variable

tmp <- lapply(names(data), function(x) 
  ggplot(data = data[, x, drop = FALSE]) + 
    aes_string(x) + xlab(x) + ylab(""))
gd <- geom_density(adjust = 2, fill = gray(0.5))
gh <- function(width) {geom_histogram(binwidth = width)}

#Arrange grids
grid.arrange(
  tmp[[2]] + gd,
  tmp[[15]] + gd,
  tmp[[28]] + gd,
  tmp[[41]] + gd,
  tmp[[54]] + gd,
  tmp[[67]] + gd,
  nrow = 2
)
## Scale
data_scaled <- data
tmp <- c("TOTUSJH_dx", "TOTUSJH_dx2")
data_scaled[tmp] <- as.data.frame(sign(data[tmp]) * abs(data[tmp])^(1/2))
tmp <- c("TOTUSJH_last", "TOTUSJH_max","TOTUSJH_dw","TOTUSJH_dw2")
data_scaled[tmp] <- as.data.frame(sign(data[tmp]) * abs(data[tmp])^(1/5))

#Plot again
tmp <- lapply(names(data_scaled), function(x) 
  ggplot(data = data_scaled[, x, drop = FALSE]) + 
    aes_string(x) + xlab(x) + ylab(""))
gd <- geom_density(adjust = 2, fill = gray(0.5))
gh <- function(width) {geom_histogram(binwidth = width)}
grid.arrange(
  tmp[[2]] + gd,
  tmp[[15]] + gd,
  tmp[[28]] + gd,
  tmp[[41]] + gd,
  tmp[[54]] + gd,
  tmp[[67]] + gd,
  nrow = 2
)

#### Kigger på step BIC
form <- 
  label ~ TOTUSJH_dx*TOTUSJH_dx2 + TOTUSJH_dw*TOTUSJH_dw2 + TOTUSJH_max + TOTUSJH_last

nulModel <- glm(label ~ 1, data = data_scaled, family = binomial(link = "logit"))
fullModel <- glm(form, data = data_scaled, family = binomial(link = "logit"))
summary(fullModel)
#add1(nulModel, form,test = "LRT")

n_samples <- nrow(data_scaled)
forward_BIC <- step(nulModel, 
                    direction = "forward", 
                    k = log(n_samples), #BIC
                    scope = list(upper = fullModel),
                    trace = 0 #How much information to print
                    )
summary(forward_BIC) #Last og dx er de eneste relevante


### Scale dx'er korrekt, lav forward BIC op til final model men nu med dx

### Scaling ------
tmp <- lapply(names(data), function(x) 
  ggplot(data = data[, x, drop = FALSE]) + 
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

############ FORWARD BIC

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

# Hmm... dx'erne ser desvære ret ligegyldige ud
n_samples <- nrow(data_scaled)
forward_BIC <- step(nulModel, 
                    direction = "forward", 
                    k = log(n_samples), #BIC
                    scope = list(upper = fullModel),
                    trace = 0 #How much information to print
)
summary(forward_BIC) #Saaatan, kun en enkelt dx desværre..

forward_back <- step(forward_BIC, 
                     direction = "backward", 
                     k = log(n_samples), #BIC
                     trace = 0 #How much information to print
)
summary(forward_back)

#final_form <- label ~ TOTUSJH_last + XR_MAX_last*NA_XR_MAX_last + R_VALUE_last + 
#  TOTPOT_last + MEANGBT_last + MEANGBZ_last + MEANGAM_last + SAVNCPP_last
final_form <- forward_back$formula
final_model <- glm(final_form, data = data_scaled, family = binomial(link = "logit"))
summary(final_model)

################### TJEK AF LINEARITET ###################
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
  qplot(XR_MAX_dx, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(MEANGAM_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(SAVNCPP_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  nrow = 3
) 

# Det ser egentligt ret fint ud. Beholder denne som sidste model. 
# Nu er det tid til at kigge lidt på predictions for at se om det overhovedet dur til noget
#### Create test_set
test_set <- read.table("/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/fold2_subset.csv",
                   sep = ",",
                   header = TRUE)
test_set <- test_set[,!(names(test_set) %in% c('X','id'))] #Remove stupid ID
test_set <- test_set[1:10000,] #Kun subset til at kigge på i startenª

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
correct <- test_set_scaled$label

plot_matrix <- f1_scores_plot(predictions,correct, 100)
max(plot_matrix[,2])


