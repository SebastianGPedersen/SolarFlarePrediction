from _0_DataCreation.Read_Data import load_dataframe
from Scoring.scoring_func import f1_scores_plot
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from General.Paths import Gitlab_Path
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import quantile_transform

fold1_df = load_dataframe(filename = 'fold1_NA_pca_features.dat')
fold2_df = load_dataframe(filename = 'fold2_NA_pca_features.dat')
fold3_df = load_dataframe(filename = 'fold3_NA_pca_features.dat')
testset_df = load_dataframe(filename = 'testSet_NA_pca_features.dat')

del fold1_df['id'],fold2_df['id'],fold3_df['id'],testset_df['id']

pca_cols = ['pca_' + str(i) + '_last' for i in range(1,24)] #Maks er pca_23_last
subset_cols = pca_cols + ['R_VALUE_last', 'XR_MAX_last', 'NA_satellite_last', 'NA_SHARPmask_last', 'NA_Rmask_last', 'NA_XR_MAX_last']
              

### Quantile transform the rest
the_rest = set(fold1_df.columns) - set(subset_cols) - set(['label'])
all_rest = pd.concat([fold1_df[the_rest],fold2_df[the_rest],fold3_df[the_rest],testset_df[the_rest]])
all_rest_q = pd.DataFrame(quantile_transform(all_rest,n_quantiles = 5000),columns = all_rest.columns)


fold1_rest = all_rest_q.iloc[0:76773]
fold2_rest = all_rest_q.iloc[76773:(76773+92481)].reset_index()
fold3_rest = all_rest_q.iloc[(76773+92481):(76773+92481+27006)].reset_index()
testset_rest = all_rest_q.iloc[(76773+92481+27006):(76773+92481+27006+173512)].reset_index()

del fold2_rest['index'],fold3_rest['index'],testset_rest['index']

### Put back in
fold1_subset = fold1_df[subset_cols]
fold2_subset = fold2_df[subset_cols]
fold3_subset = fold3_df[subset_cols]
testset_subset = testset_df[subset_cols]

fold1_full = pd.concat([fold1_subset,fold1_rest],axis = 1)
fold2_full = pd.concat([fold2_subset,fold2_rest],axis = 1)
fold3_full = pd.concat([fold3_subset,fold3_rest],axis = 1)
testset_full = pd.concat([testset_subset,testset_rest],axis = 1)

del all_rest, all_rest_q, fold1_rest, fold2_rest, fold3_rest, testset_rest

## Lav logistisk regression fra R
Cs = [0.001, 0.01, 0.1,1,10, 100]
f1_list = []

for C in Cs:
    print(C)
    LR = LogisticRegression(penalty = 'l2',max_iter = 500, C = 0.005,solver = 'saga')
    LR.fit(fold1_full,fold1_df['label'])
    
    #Score on fold2
    preds = LR.predict_proba(fold2_full)[:,1]
    true_values = fold2_df['label']
    
    ## Calculate f1_score
    classes = np.zeros(len(preds),dtype = int)
    classes[preds > 0.35] = 1
    #f1_score(y_true = true_values, y_pred = classes)
    
    f1_list.append(f1_score(y_true = true_values, y_pred = classes))

#Plot the thing
plt.scatter(np.arange(len(f1_list)),f1_list)

#Ser ud til 0.005 er bedst for den fulde
LR = LogisticRegression(penalty = 'l2',max_iter = 500, C = 0.005,solver = 'saga')
train = pd.concat([fold1_full,fold2_full],axis = 0)
label = pd.concat([fold1_df,fold2_df],axis = 0)['label']
LR.fit(train,label)

#Fit on fold3 with resize and extract best score
my_preds = LR.predict_proba(fold3_full)[:,1]
true_vals = fold3_df['label']
df, best_index = f1_scores_plot(my_preds,true_vals,resize = False)
best_threshold = df['threshold'][best_index]

del train, label

### Fit on everything and predict on test
all_sets = pd.concat([fold1_full,fold2_full,fold3_full])
all_labels = pd.concat([fold1_df,fold2_df,fold3_df])['label']
LR = LogisticRegression(penalty = 'l2',max_iter = 2000, C = 0.005,solver = 'saga')
LR.fit(all_sets,all_labels)
my_preds = LR.predict_proba(testset_full)[:,1]

my_df = pd.DataFrame({'Id':np.arange(1,len(my_preds)+1),'ClassLabel':my_preds})
my_df.to_csv(Gitlab_Path + '/Models/Logistic m. penalty/ridge_ftw_probs.csv',index = False)


#Save the classes
classes = np.zeros(len(my_preds),dtype = int)
classes[my_preds > 0.33] = 1 #For at den bliver som forrest og meatchopper

my_df = pd.DataFrame({'Id':np.arange(1,len(classes)+1),'ClassLabel':classes})
my_df.to_csv(Gitlab_Path + '/Ranking/ridge_ftw_test.csv',index = False)

