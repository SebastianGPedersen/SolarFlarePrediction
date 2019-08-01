from typing import List
import time
import pickle
import numpy as np
import os
from _0_DataCreation.Read_Data import load_data
from General.Paths import Data_Path
from sklearn.linear_model import LinearRegression
from _0_DataCreation.Raw_Data_Transformations import scale_df
## This class extracts features from timeseries and saves a 2d object 

class Feature_extractor():
  
  #Name and the belonging function
  FEATURES = {'dx'  : '_calc_dx',
              'dx2' : '_calc_dx_squared',
              'dw'  : '_calc_dw',
              'dw2' : '_calc_dw_squared',
              'last': '_calc_last',
              'mean': '_calc_mean',
              'max' : '_calc_max',
              'min' : '_calc_min'}
  
  #Number of lines (only used in print statements)
  n_lines = {'fold1_NA': 76773,
             'fold2_NA': 92481,
             'fold3_NA': 27006,
             'testSet_NA': 173512}
  
  def __init__(self,
               variables: List,                                #Which variables to be extracted
               datasets: List,                                 #Which datasets (fold1, fold2, fold3, testSet)
               save_path: str = '',                            #Where in the Data folder it should be saved
               features: List = ['dx','dx2','dw','dw2', 'max','min', 'last']):    #Which features to calculate
    
    self.variables = variables
    self.datasets  = datasets
    self.save_path = Data_Path + save_path
    self.features  = features
    self.header    = None
    self.NA_threshold = 0.8
    
    #Check the path exists
    if not os.path.isdir(self.save_path):
      raise Exception("The path ''" + str(self.save_path) + "' does not exist")


  def extract_features_training(self):
    '''
    This function creates the dataset using the 'load_data' function from 'Data.Read_Data'.
    It operates line-by-line so should be able to handle unlimited number of lines in data files.
    '''
    p0 = time.time()
    
    ### Calculate features one id at a time
    for dataset in self.datasets:
      load_file_path = Data_Path + '/' + dataset + '.dat'
      save_file_path = self.save_path + '/' + dataset + '_all_last.dat'
      
      #Delete existing
      print('save path: ' + save_file_path)
      if os.path.exists(save_file_path):
          os.remove(save_file_path)
          
      
      with open(save_file_path,'wb') as save_file:
        
        ##Create the header
        header = ['id','label']
        for feature in self.features:
          header += [variable + '_' + feature for variable in self.variables]
        self.header = header
        
        pickle.dump(header,save_file)
        
        ##Calculate data
        for id, label, data in load_data(filename = load_file_path, max_row = -1):
            
            #Extract subset
            data = np.array(scale_df(data)[self.variables])
              
            #Create data_point
            data_point = [id,label]
            for feature in self.features:
              feature_func = getattr(self, self.FEATURES[feature])
              data_point += list(feature_func(xs = data))

            #save data_point
            pickle.dump(data_point,save_file)
            #save_file.write(str(data_point[1:-1]))
            
            if (id % 5000) == 0:
              print('Dataset: ' + dataset + ', Line: ' + str(id) + ' out of ' + str(self.n_lines[dataset]) + ', Time: ' + str(int(time.time()-p0)) + 's')

  def extract_features_test(self):
    '''
    This function creates the dataset using the 'load_data' function from 'Data.Read_Data'.
    It operates line-by-line so should be able to handle unlimited number of lines in data files.
    '''
    p0 = time.time()
    
    ### Calculate features one id at a time
    for dataset in self.datasets:
      load_file_path = Data_Path + '/' + dataset + '.dat'
      save_file_path = self.save_path + '/' + dataset + '_all_last.dat'
      
      #Delete existing
      print('save path: ' + save_file_path)
      if os.path.exists(save_file_path):
          os.remove(save_file_path)
          
      
      with open(save_file_path,'wb') as save_file:
        
        ##Create the header
        header = ['id']
        for feature in self.features:
          header += [variable + '_' + feature for variable in self.variables]
        self.header = header
        
        pickle.dump(header,save_file)
        
        ##Calculate data
        for id, data in load_data(filename = load_file_path, max_row = -1):
            
            #Extract subset
            data = np.array(scale_df(data)[self.variables])
              
            #Create data_point
            data_point = [id]
            for feature in self.features:
              feature_func = getattr(self, self.FEATURES[feature])
              data_point += list(feature_func(xs = data))

            #save data_point
            pickle.dump(data_point,save_file)
            #save_file.write(str(data_point[1:-1]))
            
            if (id % 5000) == 0:
              print('Dataset: ' + dataset + ', Line: ' + str(id) + ' out of ' + str(self.n_lines[dataset]) + ', Time: ' + str(int(time.time()-p0)) + 's')


  def _calc_dx(self,xs: np.ndarray) -> float:
    if len(xs.shape) == 1:
      xs = np.reshape(xs,(-1,1))
    ys  = np.arange(len(xs))
    reg = LinearRegression().fit(xs, ys)
    return reg.coef_

  def _calc_dx_squared(self,xs: np.ndarray) -> float:
    if len(xs.shape) == 1:
      xs = np.reshape(xs,(-1,1))
    dxs = xs[1:,:]-xs[:-1,:]
    ys  = np.arange(len(dxs))
    reg = LinearRegression().fit(dxs, ys)
    return reg.coef_

  def _calc_dw(self,xs: np.ndarray) -> float:
    if len(xs.shape) == 1:
      xs = np.reshape(xs,(-1,1))
    dW = np.sum(np.square(xs[1:,:]-xs[:-1,:]),axis = 0) / (xs.shape[0]-1)
    return dW

  def _calc_dw_squared(self,xs: np.ndarray) -> float:
    if len(xs.shape) == 1:
      xs = np.reshape(xs,(-1,1))
    dws = np.square(xs[1:]-xs[:-1])
    ys  = np.arange(len(dws))
    reg = LinearRegression().fit(dws, ys)
    return reg.coef_

  def _calc_last(self,xs: np.ndarray) -> float:
    if len(xs.shape) == 1:
      xs = np.reshape(xs,(-1,1))
    return xs[-1,:]

  def _calc_mean(self,xs: np.ndarray) -> float:
    if len(xs.shape) == 1:
      xs = np.reshape(xs,(-1,1))
    return np.mean(xs,axis = 0)
  
  def _calc_max(selx,xs: np.ndarray) -> float:
    if len(xs.shape) == 1:
      xs = np.reshape(xs,(-1,1))
    return np.max(xs,axis = 0)    

  def _calc_min(selx,xs: np.ndarray) -> float:
    if len(xs.shape) == 1:
      xs = np.reshape(xs,(-1,1))
    return np.min(xs,axis = 0)   


#Take the 13 used variables from article. 'AREA_ACR' is not in the data set but 'XR_MAX' is instead.
if __name__ == '__main__':

  variables = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', \
              'TOTFZ', 'MEANPOT', 'EPSZ', 'SHRGT45', 'R_VALUE', 'XR_MAX', 'NA_satellite', \
              'NA_SHARPmask', 'NA_Rmask', 'NA_XR_MAX']
  extra_vars = ['MEANSHR', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANJZH','MEANGBH', 'TOTFY', 'MEANJZD', 'MEANALP', \
                'TOTFX', 'EPSY', 'EPSX']
  all_vars = variables + extra_vars
  features = ['last']
  
  #Training sets
  datasets = ['fold1_NA','fold2_NA','fold3_NA']
  Feature_extractor(variables = all_vars,
                    datasets  = datasets,
                    features  = features).extract_features_training()
  
  #Testsets
  datasets = ['testSet_NA']
  Feature_extractor(variables = all_vars,
                    datasets  = datasets,
                    features  = features).extract_features_test()
