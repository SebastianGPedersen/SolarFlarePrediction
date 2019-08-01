from _0_DataCreation.Read_Data import load_dataframe
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from time import time

fold1_df = load_dataframe(filename = 'fold1_NA_features.dat')
fold2_df = load_dataframe(filename = 'fold2_NA_features.dat')

del fold1_df['id']
del fold2_df['id']

#fold1_df = fold1_df[['label','pca_1_last','pca_2_last','pca_3_last','pca_4_last','pca_5_last','pca_6_last','pca_7_last','pca_8_last','pca_9_last','pca_10_last']]
#fold2_df = fold2_df[['label','pca_1_last','pca_2_last','pca_3_last','pca_4_last','pca_5_last','pca_6_last','pca_7_last','pca_8_last','pca_9_last','pca_10_last']]

## Create the different sets
one_rows = fold1_df.loc[(fold1_df['label'] == 1),:]
zero_rows = fold1_df.loc[(fold1_df['label'] == 0),:]

##create train_sets
n_rfs = int(np.ceil(len(zero_rows) / len(one_rows))) #We make 6 regressors
n_samples = int(round(len(zero_rows)/n_rfs,0))
trains = [zero_rows.iloc[n_samples * (x-1):n_samples * x,:] for x in range(1,n_rfs)]
trains += [zero_rows.iloc[n_samples*(n_rfs-1):,:]]


## Take equal amount of one rows and zero rows, and train random forrests
n_features = int(len(fold1_df.columns) / 4)

predictions = np.zeros((len(fold2_df),n_rfs))
p0 = time()
for i in range(n_rfs):
    print(i)
    my_train = pd.concat([one_rows,trains[i]])
    clf = RandomForestRegressor(n_estimators = 10,max_features = n_features)
    clf.fit(my_train.iloc[:,1:],my_train.iloc[:,0])
    predictions[:,i] = clf.predict(fold2_df.iloc[:,1:])
print(time()-p0)


## Ensemble the predictions
preds_ens = np.mean(predictions,axis = 1)

my_max = 0
for p in range(1,99):
    pred_classes = np.zeros(len(fold2_df))
    pred_classes[preds_ens > p/100] = 1
    true_vals = fold2_df['label']
    
    TP = sum(((true_vals == 1) & (pred_classes == 1)))
    FP = sum(((true_vals == 0) & (pred_classes == 1)))
    FN = sum(((true_vals == 1) & (pred_classes == 0)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    new = 2 * precision * recall / (precision + recall) #Den kan kun få 0.6 med all features, stramt
    if new > my_max:
        print('p: ' + str(p))
        print('new: ' + str(round(new,4)))
        my_max = new


