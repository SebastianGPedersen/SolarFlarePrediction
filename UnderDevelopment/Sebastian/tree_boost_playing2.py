from _0_DataCreation.Read_Data import load_dataframe
from sklearn.ensemble import GradientBoostingRegressor
from Scoring.scoring_func import f1_scores_plot
import numpy as np

fold1_df = load_dataframe(filename = 'fold1_NA_features.dat')
fold2_df = load_dataframe(filename = 'fold2_NA_features.dat')

del fold1_df['id']
del fold2_df['id']

n_features = int(len(fold1_df.columns) / 4)

clf = GradientBoostingRegressor('ls',learning_rate = 0.05, n_estimators = 200,max_features = n_features)
clf.fit(fold1_df.iloc[:,1:],fold1_df.iloc[:,0])
preds_ens = clf.predict(fold2_df.iloc[:,1:]) 

## Ensemble the predictions
true_values = fold2_df['label']
df, best_index = f1_scores_plot(preds_ens,true_values)
df['f1_score'][best_index] #Li


### Check perfomance on fold3
fold3_df = load_dataframe(filename = 'fold3_NA_features.dat')
del fold3_df['id']
dw_cols = [x for x in fold1_df.columns if x[-2:] == 'dw' and x[:3] == 'pca']
fold3_df[dw_cols] = np.log(np.array(fold3_df[dw_cols]))
fold3_df = fold3_df.replace([-np.inf],0)
x_test = fold3_df[[x for x in fold3_df.columns if x != 'label']]
y_test = fold3_df['label']
my_preds = clf.predict(x_test)



#_, best_index = f1_scores_plot(my_preds,y_test,resize = False) #0.712
df, best_index = f1_scores_plot(my_preds,y_test,resize = True) #0.653
best_threshold = df['threshold'][best_index]





'''
###### Lidt hygge test
classes = np.zeros(len(fold2_df))
classes[preds_ens > df['threshold'][best_index]] = 1

TP = sum((classes == 1) & (true_values == 1 ))
FP = sum((classes == 1) & (true_values == 0 ))    
TN = sum((classes == 0) & (true_values == 0)) 
FN = sum((classes == 0) & (true_values == 1))    

old_f1 = df['f1_score'][best_index] #Li
new_f1 = df['f1_score'][best_index] #Li

i = 0
while new_f1 < old_f1 + 0.003:
    i+=1
    TP -= 1
    TN += 1
    FN -= 1
    TP += 1
    prec = TP / (TP + FP)
    recall = TP / (TP + FN)
    new_f1 = 2 * prec * recall / ( prec + recall)
    
print(int(i*2))  #Der skal rykkes 260 (130 hver vej), dvs. faktisk ca. 0.3%
260 / len(fold2_df)
'''
    
    