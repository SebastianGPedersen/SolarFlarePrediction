# Loading some libraries
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, LSTM, Conv1D, Dense
from tensorflow.keras import backend as K

from General.Paths import Data_Path
from _0_DataCreation.Read_Data import batch_generator

# Defining the model architechture
def lstm_model(time_steps=60, num_features=16, filters=10,
               kernel_size=10, strides=5):
    
    inputs = Input(shape=(time_steps, num_features))
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides)(inputs)
    x = LSTM(units=32, activation='relu', return_sequences=True)(x)
    x = LSTM(units=16, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



if __name__ == '__main__':
    
    batch_size = 32
    time_steps = 60
    num_features = 10
    
    my_model = lstm_model(num_features=num_features)
    
    opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)
    
    my_model.compile(optimizer=opt,
                     loss='sparse_categorical_crossentropy',
                     metrics=[f1])
    
    # Setting up the data streaming
    train_gen = batch_generator(filename=Data_Path + '/fold1_NA.dat',
                                batch_size=batch_size,
                                time_steps=time_steps,
                                num_features=num_features)
    
    valid_gen = batch_generator(filename=Data_Path + '/fold2_NA.dat',
                                batch_size=batch_size,
                                time_steps=time_steps,
                                num_features=num_features)
    
    class_weight = {0: 0.3,
                    1: 0.7}
    
    
    
    my_model.fit_generator(generator=train_gen,
                           validation_data=valid_gen,
                           validation_steps=100,
                           steps_per_epoch=1000,
                           epochs=10,
                           class_weight=class_weight)
    
    X, _ = next(valid_gen)
    
    pred = my_model.predict_on_batch(X)
    classes = pred.argmax(axis=-1)
    np.mean(classes)
    
    
    
    
    











