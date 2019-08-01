import pandas as pd
import numpy as np
from General.Paths import Gitlab_Path


df = pd.read_csv(Gitlab_Path + '/Models/Ensemble/Ensemble_NN_rigde_classes.csv',sep = ";")

my_df = pd.DataFrame({'Id':np.arange(1,len(df)+1),'ClassLabel':df['Ensemble'].astype(int)})
my_df.to_csv(Gitlab_Path + '/Ranking/Ensemble_NN_rigde_classes.csv',index = False)

