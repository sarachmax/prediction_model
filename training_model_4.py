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
import matplotlib.pyplot as plt 
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
training_set = dataset.iloc[:, 2:6].values

from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler(feature_range = (0,1))
training_set = sc.fit_transform(training_set)

# create indicator RSI 
# from 2000.01.01 to 2015.12.31
training_set = np.array(training_set)

X_train = []
y_train = [] 
look_back_day = 22
for i in range(look_back_day , 4161):
    X_train.append(training_set[i-look_back_day:i, :])
    y_train.append(training_set[i, 0])
X_train , y_train = np.array(X_train) , np.array(y_train)

#########################################################################################
# RNN 
#########################################################################################

from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout 

# Initialising the RNN 
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 128, return_sequences = True, input_shape =(X_train.shape[1],X_train.shape[2])))
regressor.add(Dropout(0.2)) # Dropout 20% of layers recommended   
 
# Adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 64))
regressor.add(Dropout(0.2))

# Adding the output layer 
regressor.add(Dense(units = 16, init = 'uniform', activation='relu'))
regressor.add(Dense(units = 1, init='uniform', activation='linear')) 

# Compiling the RNN 
regressor.compile(optimizer = 'RMSprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set 
regressor.fit(X_train, y_train, epochs = 500, batch_size = 32)

#########################################################################################
# save/load prediction model  
#########################################################################################

# save model to json file
model_json = regressor.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
regressor.save_weights('model.h5')
print('saved model')
print('traning done')

"""
# load model from json file 
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
print('loaded model')

#eveluate loaded model on test data 
loaded_model.compile(optimizer = 'RMSprop', loss = 'mean_squared_error')

#########################################################################################
# Prediction Result 
#########################################################################################
# init input index for test_set 
test_day = 300
test_result_day = look_back_day + test_day     
start_input_index = end_index - look_back_day 
end_input_index = start_input_index + test_result_day

inputs = dataset_import.iloc[start_input_index:end_input_index, 2:6].values
inputs = sc.transform(inputs)

X_test = []
for i in range(look_back_day, test_result_day): 
    X_test.append(inputs[i-look_back_day:i, :])
X_test = np.array(X_test)

predicted_close_price = loaded_model.predict(X_test)
predicted_close_price = sc.inverse_transform(predicted_close_price)

real_close_price = dataset_import.iloc[end_index:end_input_index, 5:6].values
copy_close_price = real_close_price

#########################################################################################
# Visualization
#########################################################################################
print('----------------------------------------------------------------------------------')
print('---------------------------- Predict in :', test_day , 'days ------------------------------')
print('----------------------------------------------------------------------------------')
plt.plot(real_close_price, color = 'red', label = 'Real Price')
plt.plot(predicted_close_price, color = 'blue', label = 'Predicted Price')
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

#########################################################################################
# Accuracy check 
#########################################################################################

accuracy = 0
direction = [] 
predict_period = predicted_close_price.shape[0]
for i in range(1, predict_period):
    true_direction = real_close_price[i] - real_close_price[i-1]
    predict_direction = predicted_close_price[i] - real_close_price[i-1]
    if true_direction < 0 and predict_direction < 0 : 
        accuracy += 1
    elif true_direction > 0 and predict_direction > 0:
        accuracy += 1    
    if predict_direction > 0 : 
        direction.append([1])
    elif predict_direction < 0 : 
        direction.append([-1])
    else :
        direction.append([0])
accuracy = accuracy/predict_period * 100
print('accuracy :', accuracy ,'%')





    
    
    



