### READ DATA ------------------------------------------------------------------------------------
fold1 <- read.table("/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/fold1_NA_features.csv",
                   sep = ",",
                   header = TRUE)
fold1 <- fold1[1:10000,]
fold1 <- fold1[,!(names(fold1) %in% c('X','id'))] #Remove stupid ID
data <- fold1[,c("label", "pca_1_last", "pca_2_last", "pca_3_last", "pca_4_last", "pca_5_last", "pca_6_last", "pca_7_last", "pca_8_last", "pca_9_last", "pca_10_last", "XR_MAX_last", "R_VALUE_last", 
                 "NA_satellite_last", "NA_SHARPmask_last", "NA_Rmask_last", "NA_XR_MAX_last")]

my_form <- label ~ pca_1_last + pca_2_last + pca_3_last + pca_4_last + pca_5_last + pca_6_last + pca_7_last + pca_8_last + pca_9_last + pca_10_last + XR_MAX_last + R_VALUE_last + 
  NA_satellite_last + NA_SHARPmask_last + NA_Rmask_last + NA_XR_MAX_last
my_form <- label ~ pca_1_last

final_model <- glm(my_form, data = data, family = binomial(link = "logit"))

##import 
fold2 <- read.table("/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/fold2_NA_features.csv",
                    sep = ",",
                    header = TRUE)
fold2 <- fold2[,!(names(fold2) %in% c('X','id'))] #Remove stupid ID
test_set <- fold2[,c("label", "pca_1_last", "pca_2_last", "pca_3_last", "pca_4_last", "pca_5_last", "pca_6_last", "pca_7_last", "pca_8_last", "pca_9_last", "pca_10_last", "XR_MAX_last", "R_VALUE_last", 
                     "NA_satellite_last", "NA_SHARPmask_last", "NA_Rmask_last", "NA_XR_MAX_last")]

predictions <- predict(final_model, newdata = test_set, type = 'response')
true_vals <- test_set$label
plot_matrix <- f1_scores_plot(predictions,true_vals, 100)
max(plot_matrix[,2])

## Check fold3
fold3 <- read.table("/Users/SebastianGPedersen/Google Drive/LargeScale/Processed_Data/Sebastian/fold3_NA_features.csv",
                    sep = ",",
                    header = TRUE)
fold3 <- fold3[,!(names(fold2) %in% c('X','id'))] #Remove stupid ID
test_set2 <- fold3[,c("label", "pca_1_last", "pca_2_last", "pca_3_last", "pca_4_last", "pca_5_last", "pca_6_last", "pca_7_last", "pca_8_last", "pca_9_last", "pca_10_last", "XR_MAX_last", "R_VALUE_last", 
                     "NA_satellite_last", "NA_SHARPmask_last", "NA_Rmask_last", "NA_XR_MAX_last")]

predictions <- predict(final_model, newdata = test_set2, type = 'response')
true_vals <- test_set2$label
plot_matrix <- f1_scores_plot(predictions,true_vals, 100)
max(plot_matrix[,2])


## Det her er nærmest RF, hvis 4% uafhængig af tidligere logit, drop meatchopper..?

