from _0_DataCreation.Read_Data import load_dataframe
from Scoring.scoring_func import f1_scores_plot
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from General.Paths import Gitlab_Path

fold1_df = load_dataframe(filename = 'fold1_NA_features.dat')
fold2_df = load_dataframe(filename = 'fold2_NA_features.dat')
fold3_df = load_dataframe(filename = 'fold3_NA_features.dat')
testset_df = load_dataframe(filename = 'testSet_NA_features.dat')

subset_cols = ['pca_1_last', 'pca_2_last','pca_3_last','pca_4_last','pca_5_last', 'pca_6_last','pca_7_last','pca_8_last','pca_9_last','pca_10_last', \
               'R_VALUE_last', 'XR_MAX_last', 'NA_satellite_last', 'NA_SHARPmask_last', 'NA_Rmask_last', 'NA_XR_MAX_last']
                                      
fold1_subset = fold1_df[subset_cols]
fold2_subset = fold2_df[subset_cols]
fold3_subset = fold3_df[subset_cols]
testset_subset = testset_df[subset_cols]

## Lav logistisk regression fra R
LR = LogisticRegression()
LR.fit(fold1_subset,fold1_df['label'])

#Score on f2
my_preds = LR.predict_proba(fold2_subset)[:,1]
true_vals = fold2_df['label']
temp = f1_scores_plot(my_preds,true_vals) #Næsten det samme. Lidt under

#Fit on fold3 with resize and extract best score
my_preds = LR.predict_proba(fold3_subset)[:,1]
true_vals = fold3_df['label']
df, best_index = f1_scores_plot(my_preds,true_vals,resize = True) #0.653
best_threshold = df['threshold'][best_index]


### Fit on everything and predict on test
all_sets = pd.concat([fold1_subset,fold2_subset,fold3_subset])
all_labels = pd.concat([fold1_df,fold2_df,fold3_df])['label']
LR = LogisticRegression()
LR.fit(fold1_subset,fold1_df['label'])
my_preds = LR.predict_proba(testset_subset)[:,1]

#Save the classes
classes = np.zeros(len(my_preds),dtype = int)
classes[my_preds > 0.35] = 1 #For at den bliver som forrest og mmeatchopper

my_df = pd.DataFrame({'Id':np.arange(1,len(classes)+1),'ClassLabel':classes})
my_df.to_csv(Gitlab_Path + '/Ranking/logistic_pt2.csv',index = False)

