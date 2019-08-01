from typing import List
import time
import pickle
import numpy as np
import pandas as pd
import os
from _0_DataCreation.Read_Data import load_data
Data_Path = '/Users/SebastianGPedersen/Dropbox/KU/6. aar/LSDA/Kaggle/Data'


## This class NA_handles binary files and saves them again

class NA_handler():

  #Number of lines (only used in print statements)
  n_lines = {'fold1': 76773,
             'fold2': 92481,
             'fold3': 27006,
             'testSet': 173512}

  def __init__(self,
               datasets: List = ['fold1'],#Which datasets (fold1, fold2, fold3, testSet)
               save_path: str = ''):

    self.datasets  = datasets
    self.save_path = Data_Path + save_path
    self.NA_threshold = 0.8
    self.nNAs = np.zeros(25) #25 variables
    self.division_variables = ['MEANPOT','EPSZ','SHRGT45','MEANSHR','MEANGAM','MEANGBT','MEANGBZ','MEANGBH','MEANJZH','MEANJZD','MEANALP', 'EPSY','EPSX']
    self.nnonzeroR_values = 0
    
    #Check the path exists
    if not os.path.isdir(self.save_path):
      raise Exception("The path ''" + str(self.save_path) + "' does not exist")


  def handle_NA_training(self):
    '''
    This function creates the dataset using the 'load_data' function from 'Data.Read_Data'.
    It operates line-by-line so should be able to handle unlimited number of lines in data files.
    '''
    p0 = time.time()

    ### Calculate features one id at a time
    for dataset in self.datasets:
      load_file_path = Data_Path + '/' + dataset + '.dat'
      save_file_path = self.save_path + '/' + dataset + '_NA.dat'

      #Delete existing
      print('save path: ' + save_file_path)
      if os.path.exists(save_file_path):
          os.remove(save_file_path)


      with open(save_file_path,'wb') as save_file:

        ##Calculate data
        for id, label, data in load_data(filename = load_file_path, max_row = -1):
            
            #There are 3 Types of NA:
            #1) There is no satelite data (all rows are NA)
            #2) The SHARP mask is empty (some NAs, the rest are zeros except XR_MAX)
            #3) The R mask is empty (R is zero)

            #For later use
            non_division_vars = data.columns.difference(['XR_MAX'] + self.division_variables)
            non_XR = data.columns.difference(['XR_MAX'])
            
            ##Check if satellite is mising (everythin but XR_MAX have NAs)
            NA_satellite_index = data.index[data[non_XR].isnull().all(1)]
            data['NA_satellite'] = 0
            data.at[NA_satellite_index,'NA_satellite'] = 1
            '''
            if len(NA_satellite_index) > 0:
                self.save_set = data
                print('missing satellite')
                return
            '''
            ##Check if no SHARPmask (self.division_variables are NaN, the rest are zero)
            indices = ((data[non_division_vars] == 0).all(axis = 1) & data[self.division_variables].isna().all(axis = 1))
            
            data['NA_SHARPmask'] = 0
            data.at[indices,'NA_SHARPmask'] = 1
            
            '''
            if len(indices) > 0:
                self.save_set = data
                print('missing sharp')
                return
            '''    
            ##Check if no Rmask
            data['NA_Rmask'] = 0
            data.at[data['R_VALUE'] == 0,'NA_Rmask'] = 1
            
            ##Check if no XR_dat
            data['NA_XR_MAX'] = 0
            data.at[data['XR_MAX'] == -99999,'NA_XR_MAX'] = 1

            ### Find rows with NA but both satellite data and SHARPmask (i think really few)
            NA_indices  = data.isna().any(axis = 'columns')
            NA_butsharp = (NA_indices & (data['NA_SHARPmask'] == 0) & (data['NA_satellite'] == 0))
            #if sum(NA_butsharp) > 0:
            #    print('id: ' + str(id) + ' har NA')
            
            ##Replace -99999 with NA in XR_MAX
            data.at[data['XR_MAX'] == -99999,'XR_MAX'] = float('NaN')
                   
            data = self._NA_linear_interpolate(data) #Replaces full NA-rows with 0
            
            #save data_point
            pickle.dump((id,label,data),save_file)
            #save_file.write(str(data_point[1:-1]))
            
            if (id % 5000) == 0:
              print('Dataset: ' + dataset + ', Line: ' + str(id) + ' out of ' + str(self.n_lines[dataset]) + ', Time: ' + str(int(time.time()-p0)) + 's')
            

  def handle_NA_test(self):
    '''
    This function creates the dataset using the 'load_data' function from 'Data.Read_Data'.
    It operates line-by-line so should be able to handle unlimited number of lines in data files.
    '''
    p0 = time.time()

    ### Calculate features one id at a time
    for dataset in self.datasets:
      load_file_path = Data_Path + '/' + dataset + '.dat'
      save_file_path = self.save_path + '/' + dataset + '_NA.dat'

      #Delete existing
      print('save path: ' + save_file_path)
      if os.path.exists(save_file_path):
          os.remove(save_file_path)


      with open(save_file_path,'wb') as save_file:

        ##Calculate data
        for id, data in load_data(filename = load_file_path, max_row = -1):
            
            #There are 3 Types of NA:
            #1) There is no satelite data (all rows are NA)
            #2) The SHARP mask is empty (some NAs, the rest are zeros except XR_MAX)
            #3) The R mask is empty (R is zero)

            #For later use
            non_division_vars = data.columns.difference(['XR_MAX'] + self.division_variables)
            non_XR = data.columns.difference(['XR_MAX'])
            
            ##Check if satellite is mising (everythin but XR_MAX have NAs)
            NA_satellite_index = data.index[data[non_XR].isnull().all(1)]
            data['NA_satellite'] = 0
            data.at[NA_satellite_index,'NA_satellite'] = 1
            '''
            if len(NA_satellite_index) > 0:
                self.save_set = data
                print('missing satellite')
                return
            '''
            ##Check if no SHARPmask (self.division_variables are NaN, the rest are zero)
            indices = ((data[non_division_vars] == 0).all(axis = 1) & data[self.division_variables].isna().all(axis = 1))
            
            data['NA_SHARPmask'] = 0
            data.at[indices,'NA_SHARPmask'] = 1
            
            '''
            if len(indices) > 0:
                self.save_set = data
                print('missing sharp')
                return
            '''
            
            ##Check if no Rmask
            data['NA_Rmask'] = 0
            data.at[data['R_VALUE'] == 0,'NA_Rmask'] = 1
            
            ##Check if no XR_dat
            data['NA_XR_MAX'] = 0
            data.at[data['XR_MAX'] == -99999,'NA_XR_MAX'] = 1

            ### Find rows with NA but both satellite data and SHARPmask (i think really few)
            NA_indices  = data.isna().any(axis = 'columns')
            NA_butsharp = (NA_indices & (data['NA_SHARPmask'] == 0) & (data['NA_satellite'] == 0))
            #if sum(NA_butsharp) > 0:
            #    print('id: ' + str(id) + ' har NA')

            ##Replace -99999 with NA in XR_MAX
            data.at[data['XR_MAX'] == -99999,'XR_MAX'] = float('NaN')
              
            data = self._NA_linear_interpolate(data) #Replaces full NA-rows with 0
            
            #save data_point
            pickle.dump((id,data),save_file)
            #save_file.write(str(data_point[1:-1]))
            
            if (id % 5000) == 0:
              print('Dataset: ' + dataset + ', Line: ' + str(id) + ' out of ' + str(self.n_lines[dataset]) + ', Time: ' + str(int(time.time()-p0)) + 's')


  def _NA_linear_interpolate(self,data: pd.DataFrame):
    
    #If whole column is NA, let it be zero
    columns_NA = data.isna().all(axis = 0).values
    data.loc[:,columns_NA] = 0
            
    #Interpolate data
    data_np = np.apply_along_axis(self._NA_row_interpolate,axis = 0, arr = np.array(data))
    data_df = pd.DataFrame(data_np,columns = data.columns)
    
    return data_df

  def _NA_row_interpolate(self,row):
      mask = np.isnan(row) # Problem entries
      # Problem data = interpolate at problem data from non-problem data
      row[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), row[~mask])
      return(row)



#Take the 13 used variables from article. 'AREA_ACR' is not in the data set but 'XR_MAX' is instead.
if __name__ == '__main__':

  #Training sets
  datasets = ['fold1','fold2','fold3']
  my_extractor = NA_handler(datasets)
  my_extractor.handle_NA_training()
    
  #Testsets
  datasets = ['testSet']
  my_extractor = NA_handler(datasets)
  my_extractor.handle_NA_test()
  
  
