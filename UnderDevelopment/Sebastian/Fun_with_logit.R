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
data <- read.table("/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/fold1_all_last.csv",
                   sep = ",",
                   header = TRUE)
data <- data[,!(names(data) %in% c('X'))] #Remove stupid ID
#data <- data[1:10000,] #Kun subset til at kigge på i startenª

## Create functions and plot

tmp <- lapply(names(data), function(x) 
  ggplot(data = data[, x, drop = FALSE]) + 
    aes_string(x) + xlab(x) + ylab(""))
gd <- geom_density(adjust = 2, fill = gray(0.5))
gh <- function(width) {geom_histogram(binwidth = width)}

#Arrange grids
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
  tmp[[12]] + gd,
  tmp[[13]] + gd,
  nrow = 3
)

grid.arrange(
  tmp[[14]] + gd,
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
  tmp[[28]] + gd,
  nrow = 3
)

# They are damn scewed

# ------------------------ KIG PÅ KORRELATIONER ------------------------------------

#### Spearman plot fordi jeg ikke har normaliseret i dette her
data_sub <- data[,!(names(data) %in% c('label', 'NA_SHARPmask_last', 'NA_Rmask_last', 'NA_satellite_last','NA_XR_MAX'))] #Remove stupid ID
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

# Holy f*ck they are correlated. Almost as if only a few actual variables
##GROUPS:
#1) TOTBSQ, TOTPOT, TOTUSJH, TOTUSJZ, USFLUX, -TOTFZ
#2) ABSNJZH, SAVNCPP
#3) EPSZ, MEANPOT, SHRGT45
#4) R_VALUE (ret tæt på den første gruppe)
#5) XR_MAX


#### SCALING
data_scaled <- data

tmp <- c("TOTUSJH_last", "TOTBSQ_last", "TOTPOT_last", "TOTUSJZ_last", 
         "ABSNJZH_last", "SAVNCPP_last", "USFLUX_last",
          "XR_MAX_last")
data_scaled[tmp] <- as.data.frame(sign(data[tmp]) * abs(data[tmp])^(1/5))

tmp <- c("TOTFZ_last","TOTFY_last", "TOTFX_last")
data_scaled[tmp] <- as.data.frame(sign(data[tmp]) * abs(data[tmp])^(1/2))

data_scaled["MEANPOT_last"] <- as.data.frame(sign(data["MEANPOT_last"]) * abs(data["MEANPOT_last"])^(1/5))

# Look at the scaled variables
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
  tmp[[11]] + gd,
  tmp[[12]] + gd,
  tmp[[13]] + gd,
  nrow = 3
)

grid.arrange(
  tmp[[14]] + gd,
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
  tmp[[28]] + gd,
  nrow = 3
)
## Det ser muligt ud at arbejde med nu

#### Looking at add1, stepwise etc. to make variable selection
form <- 
  label ~ TOTUSJH_last + TOTBSQ_last + TOTPOT_last + TOTUSJZ_last + ABSNJZH_last + SAVNCPP_last +
          USFLUX_last + TOTFZ_last + MEANPOT_last + EPSZ_last + SHRGT45_last + R_VALUE_last*NA_Rmask_last + XR_MAX_last*NA_XR_MAX_last +
          MEANSHR_last + MEANGAM_last + MEANGBT_last + MEANGBZ_last + MEANJZH_last +
          TOTFY_last + MEANJZD_last + MEANALP_last + TOTFX_last + EPSY_last + EPSX_last + NA_satellite_last
  
nulModel <- glm(label ~ 1, data = data_scaled, family = binomial(link = "logit"))
fullModel <- glm(form, data = data_scaled, family = binomial(link = "logit"))
summary(fullModel)
#add1(nulModel, form,test = "LRT")


# Det første ligner meget det man også ser i rapporten. 
# Jeg prøver at lege lidt med at lege med automatiske BIC tilføjelser og fjernelser 
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

final_form <- label ~ TOTUSJH_last + XR_MAX_last*NA_XR_MAX_last + R_VALUE_last*NA_Rmask_last + 
  TOTPOT_last + MEANGBT_last + MEANGBZ_last + MEANGAM_last + 
  TOTFZ_last + SAVNCPP_last

final_model <- glm(final_form, data = data_scaled, family = binomial(link = "logit"))

#label ~ TOTUSJH_last + XR_MAX_last + R_VALUE_last + 
#  TOTPOT_last + MEANGBT_last + MEANGBZ_last + MEANGAM_last + 
#  TOTFZ_last + SAVNCPP_last

## Sjovt nok er kun 5 ud af 8 med i deres, og så har vi også XR_MAX med. Dejligt!

## Jeg tvinger lige NA på XR_MAX og R_VALUE

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
  qplot(MEANGBT_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(MEANGBZ_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(MEANGAM_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(TOTFZ_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  qplot(SAVNCPP_last, .deviance,data = modelDiag) + stat_binhex(bins = 25) + binScale + geom_smooth(size = 1,method = 'gam'),
  nrow = 3
) 

# Det ser egentligt ret fint ud. Beholder denne som sidste model. 
# Nu er det tid til at kigge lidt på predictions for at se om det overhovedet dur til noget

#### Create test_set
test_set <- read.table("/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/fold2_all_last.csv",
                   sep = ",",
                   header = TRUE)
test_set <- test_set[,!(names(test_set) %in% c('X'))] #Remove stupid ID
#test_set <- test_set[1:10000,] #Kun subset til at kigge på i startenª

#Scale:
test_set_scaled <- test_set
tmp <- c("TOTUSJH_last", "TOTBSQ_last", "TOTPOT_last", "TOTUSJZ_last", 
         "ABSNJZH_last", "SAVNCPP_last", "USFLUX_last",
         "XR_MAX_last")
test_set_scaled[tmp] <- as.data.frame(sign(test_set[tmp]) * abs(test_set[tmp])^(1/5))
tmp <- c("TOTFZ_last","TOTFY_last", "TOTFX_last")
test_set_scaled[tmp] <- as.data.frame(sign(test_set[tmp]) * abs(test_set[tmp])^(1/2))
test_set_scaled["MEANPOT_last"] <- as.data.frame(sign(test_set["MEANPOT_last"]) * abs(test_set["MEANPOT_last"])^(1/5))


## Predict
predictions <- predict(final_model, newdata = test_set_scaled, type = 'response')
correct <- test_set_scaled$label

plot_matrix <- f1_scores_plot(predictions,correct, 100)
max(plot_matrix[,2])


