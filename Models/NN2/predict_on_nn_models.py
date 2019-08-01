from _0_DataCreation.Read_Data import load_dataframe
from General.Paths import Gitlab_Path
import tensorflow as tf
import numpy as np
import pandas as pd


fold3_df = load_dataframe(filename = 'fold3_NA_features.dat')


### Create model ----------------------------------------------------------------------
input_length = len(fold3_df.columns)-2 #not label or id

inputs = tf.keras.Input(shape = (input_length,))

L1 = tf.keras.layers.Dense(128, activation = 'sigmoid', kernel_initializer=tf.initializers.VarianceScaling(scale=0.01**2))(inputs)
L12 = tf.keras.layers.Dropout(0.5)(L1)

L2 = tf.keras.layers.Dense(128,activation = 'sigmoid', kernel_initializer=tf.initializers.VarianceScaling(scale=0.01**2))(L12)
L22 = tf.keras.layers.Dropout(0.5)(inputs)

L3 = tf.keras.layers.Dense(100,activation = 'sigmoid', kernel_initializer=tf.initializers.VarianceScaling(scale=0.01**2))(L22)
L32 = tf.keras.layers.Dropout(0.2)(L3)


conc_layer = tf.keras.layers.concatenate([inputs,L32])

output = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.initializers.VarianceScaling(scale=0.01**2))(conc_layer)
nn_model = tf.keras.Model(inputs=inputs, outputs=output)


# Define optimization algorithm
ada = tf.optimizers.Adam(lr=0.01, decay=0.00005)

# Compile model (i.e., build compute graph)
nn_model.compile(optimizer=ada,
              loss='binary_crossentropy')


### Predict ----------------------------------------------------------------------

test_set_df = load_dataframe(filename = 'testSet_NA_features.dat')

del test_set_df['id']
dw_cols = [x for x in test_set_df.columns if x[-2:] == 'dw' and x[:3] == 'pca']
test_set_df[dw_cols] = np.log(np.array(test_set_df[dw_cols]))
test_set_df = test_set_df.replace([-np.inf],0)


#Get the best model
best_model = nn_model
best_model.load_weights(Gitlab_Path + '/Models/NN2/nn_drop_overfitting_fold1.hdf5')

predictions = best_model.predict(test_set_df).flatten()
  
my_df = pd.DataFrame({'Id':np.arange(1,len(predictions)+1),'ClassLabel':predictions})
my_df.to_csv(Gitlab_Path + '/Models/NN2/overfit_on_fold1.csv',index = False)


classifications = np.zeros(len(predictions),dtype = int)
classifications[predictions > 0.35] = 1
       

my_df = pd.DataFrame({'Id':np.arange(1,len(classifications)+1),'ClassLabel':classifications})
my_df.to_csv(Gitlab_Path + '/Ranking/overfit_on_fold1_35.csv',index = False)


'''
my_df = pd.read_csv(Gitlab_Path + '/Ranking/best_boost_overfit.csv',sep = ";")
my_df.to_csv(Gitlab_Path + '/Ranking/best_boost_overfit.csv',index = False,sep = ',')
'''           
               
    

