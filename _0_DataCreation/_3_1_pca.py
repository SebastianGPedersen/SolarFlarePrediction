from typing import List
import time
import pickle
import numpy as np
import pandas as pd
import os
from _0_DataCreation.Read_Data import load_data
from _0_DataCreation.Raw_Data_Transformations import scale_df
from General.Paths import Gitlab_Path, Data_Path

## Import pca and mean/variance scaler. These are fitted for last-values
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
pca_obj = load_obj(Gitlab_Path + "/_0_DataCreation/pca_obj")
pca_columns = pca_obj['columns']
My_pca = pca_obj['pca']


scaler_obj = load_obj(Gitlab_Path + "/_0_DataCreation/scale_obj")
My_norm_scaler = scaler_obj['scaler']
scaler_columns = scaler_obj['columns']

### The class to Scale, Normalize and PCA the time-series
class Scale_Norm_PCA_transformer():
  
  #Number of lines (only used in print statements)
  n_lines = {'fold1_NA': 76773,
             'fold2_NA': 92481,
             'fold3_NA': 27006,
             'testSet_NA': 173512}
  
  def __init__(self,
               datasets: List,                                 #Which datasets (fold1, fold2, fold3, testSet)
               save_path: str = ''):                           #Where in the Data folder it should be saved
    
    self.datasets  = datasets
    self.save_path = Data_Path + save_path
    self.header    = None
    self.na_s = ['NA_satellite', 'NA_SHARPmask','NA_Rmask','NA_XR_MAX']
    self.not_in_pca = ['XR_MAX','R_VALUE']

    #Check the path exists
    if not os.path.isdir(self.save_path):
      raise Exception("The path ''" + str(self.save_path) + "' does not exist")


  def scale_norm_and_pca_train(self):
    '''
    This function creates the dataset using the 'load_data' function from 'Data.Read_Data'.
    It operates line-by-line so should be able to handle unlimited number of lines in data files.
    '''
    p0 = time.time()
    
    ### Calculate features one id at a time
    for dataset in self.datasets:
      load_file_path = Data_Path + '/' + dataset + '.dat'
      save_file_path = self.save_path + '/' + dataset + '_pca.dat'
      save_file_path2 = self.save_path + '/' + dataset + '_no_pca.dat'
      
      #Delete existing
      print('save path: ' + save_file_path)
      if os.path.exists(save_file_path):
          os.remove(save_file_path)

      #Delete existing
      print('save path: ' + save_file_path2)
      if os.path.exists(save_file_path2):
          os.remove(save_file_path2)          
      
      with open(save_file_path,'wb') as save_file, open(save_file_path2,'wb') as save_file2:
        
        ##Calculate data
        for id, label, data in load_data(filename = load_file_path, max_row = -1):
            
            #Scale
            scaled_data = scale_df(data)
            normalized_data = self._normalize(scaled_data)
            pca_data = self._pca(normalized_data)
             
            #save data_point
            pickle.dump((id,label, normalized_data),save_file2)
            pickle.dump((id,label, pca_data),save_file)
            
            if (id % 5000) == 0:
              print('Dataset: ' + dataset + ', Line: ' + str(id) + ' out of ' + str(self.n_lines[dataset]) + ', Time: ' + str(int(time.time()-p0)) + 's')

  def scale_norm_and_pca_test(self):
    '''
    This function creates the dataset using the 'load_data' function from 'Data.Read_Data'.
    It operates line-by-line so should be able to handle unlimited number of lines in data files.
    '''
    p0 = time.time()
    
    ### Calculate features one id at a time
    for dataset in self.datasets:
      load_file_path = Data_Path + '/' + dataset + '.dat'
      save_file_path = self.save_path + '/' + dataset + '_pca.dat'
      save_file_path_no_pca = self.save_path + '/' + dataset + '_no_pca.dat'
      
      #Delete existing
      print('save path: ' + save_file_path)
      if os.path.exists(save_file_path):
          os.remove(save_file_path)

      #Delete existing
      print('save path: ' + save_file_path_no_pca)
      if os.path.exists(save_file_path_no_pca):
          os.remove(save_file_path_no_pca)          
      
      with open(save_file_path,'wb') as save_file, open(save_file_path_no_pca,'wb') as save_file_no_pca:
        
        ##Calculate data
        for id, data in load_data(filename = load_file_path, max_row = -1):
            
            #Scale
            scaled_data = scale_df(data)
            normalized_data = self._normalize(scaled_data)
            pca_data = self._pca(normalized_data)
             
            #save data_point
            pickle.dump((id,normalized_data),save_file_no_pca)
            pickle.dump((id,pca_data),save_file)
            
            if (id % 5000) == 0:
              print('Dataset: ' + dataset + ', Line: ' + str(id) + ' out of ' + str(self.n_lines[dataset]) + ', Time: ' + str(int(time.time()-p0)) + 's')

  def _normalize(self,data: pd.DataFrame) -> pd.DataFrame:
      
    #Remove NA columns
    not_included = set(data.columns) - set(scaler_columns)
    data_subset = data[scaler_columns]
    
    #Use the norm_scaler from outside the class
    fold_norm = pd.DataFrame(My_norm_scaler.transform(data_subset),columns = data_subset.columns)
    #print(data_subset.columns)
    #raise Exception("stop")
    #Put NA back
    full_data = pd.concat([fold_norm,data[not_included]],axis = 1)
    
    return full_data


  def _pca(self,data: pd.DataFrame) -> pd.DataFrame:
      
    #Remove NA and not-included columns
    not_included = set(data.columns) - set(pca_columns)
    data_subset = data[pca_columns]
      
    #Use the pca_scaler from outside the class
    fold_norm = pd.DataFrame(My_pca.transform(data_subset))
    fold_norm.columns = ['pca_' + str(x) for x in range(1,My_pca.n_components_+1)]

    #Put NA and not-included back
    full_data = pd.concat([fold_norm,data[not_included]],axis = 1)
    
    return full_data


#Take the 13 used variables from article. 'AREA_ACR' is not in the data set but 'XR_MAX' is instead.
if __name__ == '__main__':

  #Training sets
  datasets = ['fold2_NA','fold3_NA']
  temp = Scale_Norm_PCA_transformer(datasets  = datasets)
  temp.scale_norm_and_pca_train()
  
  #Test set
  datasets = ['testSet_NA']
  Scale_Norm_PCA_transformer(datasets  = datasets).scale_norm_and_pca_test()

