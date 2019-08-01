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
data <- read.table("/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/fold1_NA_features.csv",
                   sep = ",",
                   header = TRUE)
data <- data[,!(names(data) %in% c('X'))] #Remove stupid ID
data <- data[1:10000,] #Kun subset til at kigge på i startenª

### Only extract dx and dx2
empty <- numeric()
for (i in 1:ncol(data)) {
  if (substr(colnames(data)[i],nchar(colnames(data)[i])-1,nchar(colnames(data)[i])) %in% c("dx","x2")) {
    print("hej")
    empty <- append(empty,i)
  }
}
data <- data[colnames(data)[empty]]

## Create functions and plot

tmp <- lapply(names(data), function(x) 
  ggplot(data = data[, x, drop = FALSE]) + 
    aes_string(x) + xlab(x) + ylab(""))
gd <- geom_density(adjust = 2, fill = gray(0.5))
gh <- function(width) {geom_histogram(binwidth = width)}

#Arrange grids
grid.arrange(
  tmp[[1]] + gd,
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
  nrow = 3
)

grid.arrange(
  tmp[[13]] + gd,
  tmp[[14]] + gd,
  tmp[[18]] + gd,
  tmp[[19]] + gd,
  tmp[[20]] + gd,
  tmp[[21]] + gd,
  tmp[[22]] + gd,
  tmp[[23]] + gd,
  tmp[[24]] + gd,
  nrow = 3
)


grid.arrange(
  tmp[[33]] + gd,
  tmp[[34]] + gd,
  tmp[[38]] + gd,
  tmp[[39]] + gd,
  tmp[[30]] + gd,
  tmp[[31]] + gd,
  tmp[[32]] + gd,
  tmp[[33]] + gd,
  tmp[[34]] + gd,
  nrow = 3
)

# They are scewed. Maybe not transform these actually..?
#### Spearman plot fordi jeg ikke har normaliseret i dette her

data_sub <- data[,!(names(data) %in% c('label', 'NA_SHARPmask_dx', 'NA_Rmask_dx', 'NA_satellite_dx','NA_XR_MAX_dx',
                                       'NA_SHARPmask_dx2', 'NA_Rmask_dx2', 'NA_satellite_dx2','NA_XR_MAX_dx2'))] 
ncol(data_sub)


data_dx <- data_sub[,1:25]
cp <- cor(data.matrix(data_dx), method = "spearman")
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

data_dx2 <- data_sub[,25:50]
cp <- cor(data.matrix(data_dx2), method = "spearman")
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
