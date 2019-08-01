import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd

def f1_scores_plot(predictions,true_values, resize = False):
    '''
    predictions: List or np.array of probabilities
    true_values: List or np.array of classifications
    resize: Whether it resize the proportion of ones in 'true_values' to fit the proportion in test set

    Output: Dataframe with thresholds and f1_scores
    
    '''
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    if resize:
        test_perc = 12.64 / 100
        val_perc = sum((true_values == 1)) / len(true_values)
        one_indices = np.nonzero((true_values == 1))[0] #False is zero
        zero_indices = np.nonzero((true_values == 0))[0] #False is zero
        
        if test_perc < val_perc: #We should remove ones
            n_total = 1 / (1-test_perc) * sum((true_values == 0))
            n_desired_ones = int(round( test_perc * n_total,0))
            my_one_indices = np.random.choice(one_indices,size = n_desired_ones, replace = False)
            all_indices = np.concatenate((my_one_indices,zero_indices))
        else: #We should remove zeros
            n_total = 1/ test_perc * sum((true_values == 1))
            n_desired_zeros = int(round( (1-test_perc) * n_total,0))
            my_zero_indices = np.random.choice(zero_indices,size = n_desired_zeros, replace = False)
            all_indices = np.concatenate((one_indices,my_zero_indices))
            
        true_values = true_values[all_indices]
        predictions = predictions[all_indices]

            
        
    k = 100 #Number of points to calculate f1_score
    
    pred_probs = np.arange(min(predictions) - 0.001,
                           max(predictions) + 0.001,
                              ((max(predictions) + 0.001) - (min(predictions) - 0.001)) /k)[:k]
    
    x_axis = pred_probs
    y_axis = np.zeros(k)
    
    for i in range(k):
        classifications = np.zeros(len(predictions))
        classifications[predictions > pred_probs[i]] = 1
        
        TP = sum((classifications == 1) & (np.array(true_values) == 1))
        
        if TP > 0:
            y_axis[i] = f1_score(y_true = true_values, y_pred = classifications)
        else:
            y_axis[i] = 0
            
    plt.scatter(x_axis,y_axis)
    plt.show()
    
    
    df = pd.DataFrame({'threshold':pred_probs,'f1_score':y_axis})

    best_index = np.argmax(y_axis)
    
    print("Best threshold: " + str(round(pred_probs[best_index],2)))
    print("Best f1_score: " + str(round(y_axis[best_index],3)))

    return df, best_index

