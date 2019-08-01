from General.Paths import Gitlab_Path
from _0_DataCreation.Read_Data import load_dataframe
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from time import time
from _0_DataCreation.Raw_Data_Transformations import scale_df

### To load and save pickle objects
import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

### Load and merge data
fold1_df = load_dataframe(filename = 'fold1_NA_all_last.dat')
fold2_df = load_dataframe(filename = 'fold2_NA_all_last.dat')
fold3_df = load_dataframe(filename = 'fold3_NA_all_last.dat')
test_df = load_dataframe(filename = 'testSet_NA_all_last.dat')
del fold1_df['label'], fold2_df['label'], fold3_df['label']

all_sets = pd.concat([fold1_df,fold2_df,fold3_df,test_df])
del fold1_df, fold2_df, fold3_df, test_df


##Extract last
last_cols = [x for x in all_sets.columns.values if x[-4:] == 'last']
fold_last = all_sets[last_cols]
del all_sets

fold_last.columns = [x[:-5] for x in fold_last.columns] #remove '_last' from name

### Normalize everything but NA ---------------------------------------------------------------
na_s = ['NA_satellite', 'NA_SHARPmask','NA_Rmask','NA_XR_MAX']
fold = fold_last.drop(na_s, axis=1, inplace=False)

My_norm_scaler = StandardScaler()
fold_norm = pd.DataFrame(My_norm_scaler.fit_transform(fold),columns = fold.columns)
norm_columns = fold.columns
scaling_obj = {'columns': norm_columns, 'scaler': My_norm_scaler}

#Remove 'XR_MAX_last' and R_VALUE from data_set
not_in_pca = ['XR_MAX','R_VALUE']
fold_norm_sharp = fold_norm.drop(not_in_pca, axis=1, inplace=False)

'''
## Explained variance, choose n_components --------------------------------------
p0 = time()
pca = PCA(n_components = None)
pca.fit(fold_norm_sharp)
temp = np.cumsum(np.array(pca.explained_variance_ratio_))
print(time()-p0) #Intet tid
#Logit included 8. I will include 10 (98.2% explained variance)
'''

### PCA with all components-------------------------------------------------------------------------
n_components = len(fold_norm_sharp.columns)
My_pca = PCA(n_components = n_components)
reduced_sharp_frame =  pd.DataFrame(My_pca.fit_transform(fold_norm_sharp))
pca_obj = {'columns': fold_norm_sharp.columns, 'pca': My_pca}

reduced_sharp_frame.columns = ['pca_' + str(x) for x in range(1,n_components+1)]

## Create full (NAs, the two normaliced and the SHARP both normalized and pca'ed)
total_last = pd.concat([fold_last[na_s].reset_index(),reduced_sharp_frame.reset_index(), fold_norm[not_in_pca].reset_index()],axis = 1)
         
save_obj(pca_obj, Gitlab_Path + "/_0_DataCreation/pca_obj")
save_obj(scaling_obj, Gitlab_Path + "/_0_DataCreation/scale_obj")


