# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 23:31:39 2018

@author: Sarach
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 19:42:22 2018

@author: Sarach
"""


import numpy as np
#import matplotlib.pyplot as plt 
import pandas as pd 

#########################################################################################
# Training Data
#########################################################################################

dataset_import = pd.read_csv('EURUSD.csv')
# init training_set index 
# use dataset from 1995.01.02 to 2015.12.31
start_index = 5832
end_index = 11245 + 1
dataset = dataset_import.iloc[start_index:end_index,:]
date = dataset.iloc[:, 0:1]
training_set = dataset.iloc[:, 5:6].values

from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler(feature_range = (0,1))
training_set = sc.fit_transform(training_set)

# create indicator RSI 
# from 2000.01.01 to 2015.12.31
training_set = np.array(training_set)

X_train = []
y_train = [] 
look_back_day = 64
for i in range(look_back_day , 4161):
    X_train.append(training_set[i-look_back_day:i, :])
    y_train.append(training_set[i, :])
X_train, y_train,  = np.array(X_train), np.array(y_train)

#########################################################################################
# RNN predict price 
#########################################################################################

from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout 

regressor = Sequential()

regressor.add(LSTM(units = 1024, return_sequences = True, input_shape =(X_train.shape[1],X_train.shape[2])))
regressor.add(Dropout(0.2))  

regressor.add(LSTM(units = 512, return_sequences = True))
regressor.add(Dropout(0.2)) 

regressor.add(LSTM(units = 256, return_sequences = True))
regressor.add(Dropout(0.2)) 

regressor.add(LSTM(units = 128, return_sequences = True))
regressor.add(Dropout(0.2)) 

regressor.add(LSTM(units = 64, return_sequences = True))
regressor.add(Dropout(0.2)) 

regressor.add(LSTM(units = 32, return_sequences = True))
regressor.add(Dropout(0.2)) 

regressor.add(LSTM(units = 16))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1, init='uniform', activation='linear')) 

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 50)

#########################################################################################
# save prediction model  
#########################################################################################
# save model to json file
model_json = regressor.to_json()
with open('model_price.json', 'w') as json_file:
    json_file.write(model_json)
regressor.save_weights('model_price.h5')
print('saved model')
print('training predict price done')