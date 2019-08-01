## This module is testing this:
# https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Loading some libraries
from General.Paths import Data_Path, Gitlab_Path
import numpy as np
import tensorflow.keras.backend as K
import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, LSTM, Conv1D, Dense, Dropout, concatenate
from Scoring.scoring_func import f1_scores_plot
from General.Paths import Data_Path

import imp
import _0_DataCreation.Read_Data
imp.reload(_0_DataCreation.Read_Data)
from _0_DataCreation.Read_Data import batch_generator, batch_generator_test, batch_generator2, load_dataframe


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    return 1 - K.mean(f1)



n_lines = {'fold1': 76773,
           'fold2': 92481,
           'fold3': 27006,
           'testSet': 173512}

data_sets = {'fold1': 'fold1_NA.dat',
           'fold2': 'fold2_NA.dat',
           'fold3': 'fold3_NA.dat',
           'testSet': 'testSet_NA.dat'}
feature_sets = {'fold1': 'fold1_NA_features.dat',
           'fold2': 'fold2_NA_features.dat',
           'fold3': 'fold3_NA_features.dat',
           'testSet': 'testSet_NA_features.dat'}

batch_size = 256
num_features = 16
# Defining the model architechture
def lstm_model(num_features=16, filters=10,
               kernel_size=10, strides=5):
    
    inputs = Input(shape=(60, num_features))
    inputs2 = Dropout(0.2)(inputs)
    
    x1 = LSTM(units=32, 
             activation='tanh', 
             return_sequences=True,
             kernel_initializer = tf.initializers.VarianceScaling(scale=0.01**2),
             recurrent_initializer = tf.initializers.VarianceScaling(scale=0.01**2),
             bias_initializer =tf.initializers.TruncatedNormal(stddev=0.01),
             dropout = 0.5,
             recurrent_dropout = 0.5)(inputs2)
    x2 = LSTM(units=32, 
             activation='tanh', 
             return_sequences=True,
             kernel_initializer = tf.initializers.VarianceScaling(scale=0.01**2),
             recurrent_initializer = tf.initializers.VarianceScaling(scale=0.01**2),
             bias_initializer =tf.initializers.TruncatedNormal(stddev=0.01),
             dropout = 0.5,
             recurrent_dropout = 0.5)(x1)
    x3 = LSTM(units=32, 
             activation='tanh',
             return_sequences=True,
             kernel_initializer = tf.initializers.VarianceScaling(scale=0.01**2),
             recurrent_initializer=tf.initializers.VarianceScaling(scale=0.01**2),
             bias_initializer=tf.initializers.TruncatedNormal(stddev=0.01),
             dropout = 0.5,
             recurrent_dropout = 0.5)(x2)
    conc_layer = concatenate([inputs2,x3])
 
    x4 = LSTM(units=16, 
             activation='tanh',
             kernel_initializer = tf.initializers.VarianceScaling(scale=0.01**2),
             recurrent_initializer=tf.initializers.VarianceScaling(scale=0.01**2),
             bias_initializer=tf.initializers.TruncatedNormal(stddev=0.01),
             dropout = 0.5,
             recurrent_dropout = 0.5)(conc_layer)
    outputs = Dense(1, activation='sigmoid')(x4)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def compile_model(savename):


  my_model = lstm_model(num_features=num_features)

  opt = keras.optimizers.Adam(lr=0.001, decay=0.00003) #Sikkert fint

  my_model.compile(optimizer=opt,
                   loss=f1_loss, metrics=['accuracy', f1])
                   #loss='binary_crossentropy')
  return my_model


def fit_model(my_model, train_names, val_name,savename, epochs = 1):
  
  # Setting up the data streaming
  filenames = [Data_Path + '/' + data_sets[train_name] for train_name in train_names]
  
  
  train_gen = batch_generator2(filenames=filenames,
                              batch_size=batch_size,
                              num_features=num_features)

  valid_gen = batch_generator(filename=Data_Path + '/' + data_sets[val_name],
                              batch_size=batch_size,
                              num_features=num_features)
  my_model.fit_generator(generator=train_gen,
                         validation_data=valid_gen,
                         steps_per_epoch = 50,
                         validation_steps= 50,
                         epochs=epochs)
  return my_model


## Create the model fitting loop
training_sets = [['fold1']]
val_set = ['fold2']

for i in range(len(training_sets)):
  savename = i
  
  #Fit model
  my_model = compile_model(savename)
  my_fitted = fit_model(my_model, training_sets[i],val_set[i],savename, epochs = 1)
  

  
predictions = []

for i in range(len(training_sets)):
  save_name = i
  
  best_model = my_fitted
  
  #Create the new generat   
  test_gen = batch_generator(filename=Data_Path + '/fold2_NA.dat',
                              batch_size=batch_size,
                              num_features=num_features)

  preds = my_model.predict_generator(test_gen,
                                     steps = np.ceil( n_lines['fold2']/ batch_size)
                                     ).flatten()
  predictions.append(preds)
  
  print('juhu')
## Make the prediction ensemble on the testSet
all_preds = np.transpose(np.array(predictions))
preds = np.mean(all_preds,axis = 1)

true_vals = load_dataframe(filename = 'fold2_NA_features.dat')['label']
preds = preds[:len(true_vals)] #Vi predicter 'np.ceil' så genstarter batchen
   
df, best_index = f1_scores_plot(preds,true_vals)

## Første optimale: p = 0.42 , score =  0.654 

##Check f1 score
