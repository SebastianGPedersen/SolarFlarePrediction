# Loading some libraries
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, LSTM, Dense
from Scoring.scoring_func import f1_scores_plot
from General.Paths import Data_Path
from _0_DataCreation.Read_Data import batch_generator, batch_generator2, load_dataframe


n_lines = {'fold1': 76773,
           'fold2': 92481,
           'fold3': 27006,
           'testSet': 173512}

# Defining the model architechture
def lstm_model(num_features=16, filters=10,
               kernel_size=10, strides=5):
    
    inputs = Input(shape=(60, num_features))
    x = LSTM(units=32, activation='tanh', return_sequences=True)(inputs)
    x = LSTM(units=16, activation='tanh')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model



if __name__ == '__main__':
    

    batch_size = 256 #Ingen grund til små batches
    num_features = 16 #The 10 pca, 4 NA, R_VALUE and XR_max
    

    my_model = lstm_model(num_features=num_features)
    
    opt = keras.optimizers.Adam(lr=0.01, decay=1e-6) #Sikkert fint
    
    my_model.compile(optimizer=opt,
                     loss='binary_crossentropy')
    
    # Setting up the data streaming
    train_gen = batch_generator2(filenames = [Data_Path + '/fold1_NA.dat', Data_Path + '/fold2_NA.dat'],
                                batch_size=batch_size,
                                num_features=num_features)
    
    valid_gen = batch_generator(filename=Data_Path + '/fold3_NA.dat',
                                batch_size=batch_size,
                                num_features=num_features)
    
    my_model.fit_generator(generator=train_gen,
                           validation_data=valid_gen,
                           steps_per_epoch = np.ceil( (n_lines['fold1'] + n_lines['fold2'])/ batch_size), #Hvornår skal den stoppe med epoc og starte næste?
                           validation_steps= np.ceil( n_lines['fold3']/ batch_size), #Hvornår skal den stoppe med epoc og starte næste?
                           epochs=3)
    
    #Create the new generat   
    valid_gen = batch_generator(filename=Data_Path + '/fold3_NA.dat',
                                batch_size=batch_size,
                                num_features=num_features)
    
    preds = my_model.predict_generator(valid_gen,
                                       steps = np.ceil( n_lines['fold3']/ batch_size)
                                       ).flatten()
    
    true_vals = load_dataframe(filename = 'fold3_NA_features.dat')['label']
    preds = preds[:len(true_vals)] #Vi predicter 'np.ceil' så genstarter batchen
    
    f1_scores_plot(preds,true_vals)
    f1_scores_plot(preds,true_vals,resize = True)
    
    











