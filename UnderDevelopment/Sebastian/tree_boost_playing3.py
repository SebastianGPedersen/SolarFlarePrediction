from _0_DataCreation.Read_Data import load_dataframe
from sklearn.ensemble import GradientBoostingClassifier
from General.Paths import Gitlab_Path
import pandas as pd
from Scoring.scoring_func import f1_scores_plot
import numpy as np
from time import time

fold1_df = load_dataframe(filename = 'fold1_NA_features.dat')
fold2_df = load_dataframe(filename = 'fold2_NA_features.dat')

del fold1_df['id']
del fold2_df['id']


n_features = int(len(fold1_df.columns) / 4)
p0 = time()
clf = GradientBoostingClassifier('deviance',
                                 learning_rate = 0.05,
                                 n_estimators = 100,
                                 max_features = n_features)

clf.fit(fold1_df.iloc[:,1:],fold1_df.iloc[:,0])
preds_ens = clf.predict_proba(fold2_df.iloc[:,1:])[:,1]
print(time()-p0)

## Ensemble the predictions
true_values = fold2_df['label']
df, best_index = f1_scores_plot(preds_ens,true_values)
df['f1_score'][best_index] #Li


### Check perfomance on fold3
fold3_df = load_dataframe(filename = 'fold3_NA_features.dat')
del fold3_df['id']
dw_cols = [x for x in fold1_df.columns if x[-2:] == 'dw' and x[:3] == 'pca']
fold3_df[dw_cols] = np.log(np.array(fold3_df[dw_cols]))
fold3_df = fold3_df.replace([-np.inf],0)
x_test = fold3_df[[x for x in fold3_df.columns if x != 'label']]
y_test = fold3_df['label']
my_preds = clf.predict_proba(x_test)[:,1]

_, best_index = f1_scores_plot(my_preds,y_test,resize = False) #0.712

df, best_index = f1_scores_plot(my_preds,y_test,resize = True) #0.653
best_threshold = df['threshold'][best_index]



####  fit on full and save
all_sets = pd.concat([fold1_df,fold2_df,fold3_df])
testset = load_dataframe(filename = 'testSet_NA_features.dat')

del testset['id']

clf.fit(all_sets.iloc[:,1:],all_sets.iloc[:,0])
preds_ens = clf.predict_proba(testset)[:,1]
classes = np.zeros(len(preds_ens),dtype = int)
classes[preds_ens > best_threshold] = 1
      

my_df = pd.DataFrame({'Id':np.arange(1,len(classes)+1),'ClassLabel':classes})
my_df.to_csv(Gitlab_Path + '/Ranking/boosted_trees.csv',index = False)

