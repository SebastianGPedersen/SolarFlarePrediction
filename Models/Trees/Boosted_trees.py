### To load and save pickle objects
import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



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
                                 learning_rate = 0.01,
                                 n_estimators = 500,
                                 max_features = n_features,
                                 validation_fraction = 0.3,
                                 n_iter_no_change = 10,
                                 #min_samples_split = 10, #default is 1
                                 max_depth = 3 #default is 3..?
                                 )

clf.fit(fold1_df.iloc[:,1:],fold1_df.iloc[:,0])
preds_ens = clf.predict_proba(fold2_df.iloc[:,1:])[:,1]
print(time()-p0)

## Ensemble the predictions
true_values = fold2_df['label']
#df, best_index = f1_scores_plot(preds_ens,true_values)
#df['f1_score'][best_index] #Li


### Check perfomance on fold3

fold3_df = load_dataframe(filename = 'fold3_NA_features.dat')
del fold3_df['id']

x_test = fold3_df[[x for x in fold3_df.columns if x != 'label']]
y_test = fold3_df['label']

my_preds = clf.predict_proba(x_test)[:,1]
_, best_index = f1_scores_plot(my_preds,y_test,resize = False) #0.712
          
                              
#save_obj(clf, Gitlab_Path + "/Models/Trees/tree_0.715")
clf = load_obj(Gitlab_Path + "/Models/Trees/tree_0.715")

####  fit on full and save
all_sets = pd.concat([fold1_df,fold2_df,fold3_df])
del fold1_df, fold2_df, fold3_df

testset = load_dataframe(filename = 'testSet_NA_features.dat')
del testset['id']

clf.fit(all_sets.iloc[:,1:],all_sets.iloc[:,0])
preds_ens = clf.predict_proba(testset)[:,1]
classes = np.zeros(len(preds_ens),dtype = int)
classes[preds_ens > 0.35] = 1

my_df = pd.DataFrame({'Id':np.arange(1,len(classes)+1),'ClassLabel':classes})
my_df.to_csv(Gitlab_Path + '/Ranking/boosted_trees_validation0.3nr2.csv',index = False)

