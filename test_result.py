# change test_day (must >= 1)  
test_day = 500
#########################################################################################

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

#########################################################################################
# Data 
#########################################################################################

dataset_import = pd.read_csv('EURUSD.csv')
# init training_set index 
# use dataset from 2000.01.01 to 2015.12.31
start_index = 7084
end_index = 11245 + 1
dataset = dataset_import.iloc[start_index:end_index,:]
look_back_day = 64
sc = MinMaxScaler(feature_range = (0,1))

# init input index for test_set 
test_result_day = look_back_day + test_day     
start_input_index = end_index - look_back_day 
end_input_index = start_input_index + test_result_day

inputs = dataset_import.iloc[start_input_index:end_input_index, 5:6].values
real_close_price = dataset_import.iloc[end_index:end_input_index , 5:6].values
#real_close_price = sc.fit_transform(real_close_price)
inputs = sc.fit_transform(inputs)

input_close_price = []
X_test = []
for i in range(look_back_day, test_result_day): 
    X_test.append(inputs[i-look_back_day:i, :])
    input_close_price.append(inputs[i,:])
X_test , input_close_price = np.array(X_test), np.array(input_close_price)

#########################################################################################
# load prediction model  
#########################################################################################

# load model from json file 
from keras.models import model_from_json
json_file = open('model_price.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model_price.h5')
print('loaded model')
#eveluate loaded model on test data 
loaded_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#########################################################################################
# Prediction Result 
#########################################################################################

predicted_close_price = loaded_model.predict(X_test)
predicted_close_price = sc.inverse_transform(predicted_close_price)

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

#########################################################################################
# Visualization
#########################################################################################
# real_close_price = sc.fit_transform(real_close_price)
# predicted_close_price = sc.fit_transform(predicted_close_price)
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
