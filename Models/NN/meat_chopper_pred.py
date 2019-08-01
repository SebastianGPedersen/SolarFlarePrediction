import numpy as np
import pandas as pd
import tensorflow as tf
from General.Paths import Gitlab_Path
from _0_DataCreation.Read_Data import load_dataframe
from Scoring.scoring_func import f1_scores_plot


##Create fold1_df (just )
fold1_df = load_dataframe(filename = 'fold1_NA_features.dat')
del fold1_df['id']
dw_cols = [x for x in fold1_df.columns if x[-2:] == 'dw' and x[:3] == 'pca']
fold1_df[dw_cols] = np.log(np.array(fold1_df[dw_cols]))


### Create model ----------------------------------------------------------------------
input_length = len(fold1_df.columns) -1 #not label
#Model
inputs = tf.keras.Input(shape = (input_length,))
n1 = tf.keras.layers.Dense(1, activation = 'sigmoid')(inputs)
#output = tf.keras.layers.Dense(1, activation='sigmoid')(n1)
nn_model = tf.keras.Model(inputs=inputs, outputs=n1)

inputs = tf.keras.Input(shape = (input_length,))
n1 = tf.keras.layers.Dense(128, activation = 'sigmoid', kernel_initializer=tf.initializers.VarianceScaling(scale=0.01**2))(inputs)
n2 = tf.keras.layers.Dense(128,activation = 'sigmoid', kernel_initializer=tf.initializers.VarianceScaling(scale=0.01**2))(n1)
n3 = tf.keras.layers.Dense(64,activation = 'sigmoid', kernel_initializer=tf.initializers.VarianceScaling(scale=0.01**2))(n2)
conc_layer = tf.keras.layers.concatenate([inputs,n3])
output = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.initializers.VarianceScaling(scale=0.01**2))(conc_layer)

# Generate a network with the same structure as the one used during training
best_model = nn_model
# Set the weights to the weights that gave the lowest validation error during training
best_model.load_weights(Gitlab_Path + '/Models/NN/model_val.hdf5')



### Check perfomance on fold3
fold3_df = load_dataframe(filename = 'fold3_NA_features.dat')
del fold3_df['id']
dw_cols = [x for x in fold1_df.columns if x[-2:] == 'dw' and x[:3] == 'pca']
fold3_df[dw_cols] = np.log(np.array(fold3_df[dw_cols]))
fold3_df = fold3_df.replace([-np.inf],0)
x_test = fold3_df[[x for x in fold3_df.columns if x != 'label']]
y_test = fold3_df['label']
my_preds = best_model.predict(x_test).flatten()

#_, best_index = f1_scores_plot(my_preds,y_test,resize = False) #0.712
df, best_index = f1_scores_plot(my_preds,y_test,resize = True) #0.653
best_threshold = df['threshold'][best_index]
   
                              
                              
## Creat predictions on 
test_set_df = load_dataframe(filename = 'testSet_NA_features.dat')
del test_set_df['id']
dw_cols = [x for x in test_set_df.columns if x[-2:] == 'dw' and x[:3] == 'pca']
test_set_df[dw_cols] = np.log(np.array(test_set_df[dw_cols]))
test_set_df = test_set_df.replace([-np.inf],0)
my_y_preds = best_model.predict(test_set_df).flatten()
classifications = np.zeros(len(my_y_preds),dtype = int)
classifications[my_y_preds > best_threshold] = 1

my_df = pd.DataFrame({'Id':np.arange(1,len(classifications)+1),'ClassLabel':classifications})
my_df.to_csv(Gitlab_Path + '/Ranking/meat_chopper.csv',index = False)
