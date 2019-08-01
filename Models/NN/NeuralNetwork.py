from _0_DataCreation.Read_Data import load_dataframe
from Scoring.scoring_func import f1_scores_plot
import tensorflow as tf
import numpy as np

### Import and reshape data ----------------------------------------------------------------------
fold1_df = load_dataframe(filename = 'fold1_NA_features.dat')
fold2_df = load_dataframe(filename = 'fold2_NA_features.dat')
del fold1_df['id']
del fold2_df['id']


##Extract the relevant form from 'Fun_with_logit4:
#label ~ pca_1_last + R_VALUE_last + pca_10_last + XR_MAX_last + 
#        pca_5_last + pca_9_last + pca_6_last + log(pca_1_dw) + pca_8_last


dw_cols = [x for x in fold1_df.columns if x[-2:] == 'dw' and x[:3] == 'pca']

np.min(np.array(fold1_df[dw_cols]))
np.min(np.array(fold2_df[dw_cols]))

fold1_df[dw_cols] = np.log(np.array(fold1_df[dw_cols]))
fold1_df.loc[(fold1_df['NA_SHARPmask_min'] == 1),dw_cols] = 0
fold2_df[dw_cols] = np.log(np.array(fold2_df[dw_cols]))

subset_cols = ['pca_1_last', 'R_VALUE_last', 'pca_10_last', 'XR_MAX_last', \
                       'pca_5_last', 'pca_9_last', 'pca_6_last', 'pca_1_dw', 'pca_8_last']
fold1_subset = fold1_df[subset_cols]
fold2_subset = fold2_df[subset_cols]
fold1_subset = fold1_subset.replace([-np.inf],0)
fold2_subset = fold2_subset.replace([-np.inf],0)




### Create model ----------------------------------------------------------------------
input_length = len(fold1_subset.columns)
#Model
inputs = tf.keras.Input(shape = (input_length,))
n1 = tf.keras.layers.Dense(1, activation = 'sigmoid')(inputs)
#output = tf.keras.layers.Dense(1, activation='sigmoid')(n1)
nn_model = tf.keras.Model(inputs=inputs, outputs=n1)



## Train model as in .ipynb ----------------------------------------------------------------------
callbacks = [
  tf.keras.callbacks.TensorBoard(log_dir='./logs/run2', update_freq='batch'),
  #tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20),
  tf.keras.callbacks.ModelCheckpoint('./logs/model_val.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
]

# Define optimization algorithm
sgd = tf.optimizers.SGD(lr=0.2)

# Compile model (i.e., build compute graph)
nn_model.compile(optimizer=sgd,
              loss='MSE')

# Training loop
nn_model.fit(x_train, y_train, batch_size=25, epochs=100, 
          validation_data=(x_val, y_val), validation_freq=1, 
          #steps_per_epoch=x_train.shape[0],
          callbacks=callbacks)




## Score
my_preds = LR.predict_proba(fold2_subset)[:,1]
true_vals = fold2_df['label']
temp = f1_scores_plot(my_preds,true_vals) #Næsten det samme. Lidt under

                     

## Skal slå 0.644
