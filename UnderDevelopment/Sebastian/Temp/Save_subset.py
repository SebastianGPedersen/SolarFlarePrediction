from _0_DataCreation.Read_Data import load_dataframe
from General.Paths import Data_Path


fold1_df = load_dataframe(filename = 'fold1_NA_features.dat')
fold1_df.to_csv(path_or_buf = Data_Path + '/Sebastian/fold1_NA_features.csv',sep = ',')

fold1_df = load_dataframe(filename = 'fold2_NA_features.dat')
fold1_df.to_csv(path_or_buf = Data_Path + '/Sebastian/fold2_NA_features.csv',sep = ',')

fold1_df = load_dataframe(filename = 'fold3_NA_features.dat')
fold1_df.to_csv(path_or_buf = Data_Path + '/Sebastian/fold3_NA_features.csv',sep = ',')

fold1_df = load_dataframe(filename = 'testSet_NA_features.dat')
fold1_df.to_csv(path_or_buf = Data_Path + '/Sebastian/testSet_NA_features.csv',sep = ',')


