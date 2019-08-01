from General.Paths import Data_Path
from _0_DataCreation.Read_Data import load_data
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

#pip install --upgrade pandas

fn = Data_Path + '/fold1_NA.dat'  # obs = 0..76773
fn2 = Data_Path + '/fold2_NA.dat'  # obs = 0..92481
fn3 = Data_Path + '/fold3_NA.dat'  # obs = 0..27006
file_paths = [fn,fn2,fn3]

#Take random file and index
file_path = random.choice(file_paths)

rand_id = random.randint(1,99)

n_ones = 10
n_zeros = 10

#Take random file and random index (remember the id's are already shuffled, so up to 100 id's should be fine)
time_series_ones = []
time_series_zeros = []
zero_count = 0
one_count = 0
begin_id = 0
while True:
    if one_count == 10 and zero_count == 10:
        break
    for id, label, data in load_data(filename=file_path, max_row=100):
        if id > begin_id:
            begin_id = id
            if label == 1 and one_count < 10:
                one_count += 1
                time_series_ones.append(data['pca_1'])
                break
            elif label == 0 and zero_count < 10:
                zero_count += 1
                time_series_zeros.append(data['pca_1'])
                break
        

time_series_ones = np.array(time_series_ones)
time_series_zeros = np.array(time_series_zeros)

plt.plot(np.arange(1, time_series_zeros.shape[1]+1) / 5, time_series_zeros.transpose(),color = 'blue')
plt.plot(np.arange(1, time_series_ones.shape[1]+1) / 5, time_series_ones.transpose(),color = 'red')
plt.show()


#Plot options
fontP = FontProperties()
fontP.set_size('x-large')

import os
os.chdir(r'/Users/SebastianGPedersen/Dropbox/KU/6. aar/LSDA')
from matplotlib.lines import Line2D


### First plot
#fig = plt.figure()
fig = plt.figure()

plt.plot(np.arange(1, time_series_zeros.shape[1]+1) / 5, 
         time_series_zeros.transpose(),
         color = 'blue')

plt.plot(np.arange(1, time_series_ones.shape[1]+1) / 5, 
         time_series_ones.transpose(),
         color = 'red')

plt.xlabel('Time in hours')
plt.ylabel('PCA 1')
plt.title('20 time series of the first PCA parameter')
legend_handles = [Line2D([0], [0], color='blue', lw=2),
                Line2D([0], [0], color='red', lw=2)]
plt.legend(legend_handles,["No solar flare",
            "At least one solar flare"],
            bbox_to_anchor = (1,0.5))
fig.savefig('First_PCA_timeseries.png',bbox_inches = 'tight')
