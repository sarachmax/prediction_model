# change test_day (must >= 1)  
test_day = 200
#########################################################################################


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
#########################################################################################
# Create indicator 
#########################################################################################
def RSI_calculate(start, end):
    indicator_RSI = []
    period = 14
    temp_close = dataset_import.iloc[:,5:6].values
    for i in range(start, end):
        chg = 0 
        lost = 0 
        gain = 0 
        avg_lost = 0
        avg_gain = 0 
        for k in range(period):
            chg = temp_close[i+k,0]-temp_close[i+k-1,0]
            if chg < 0 :
                lost += abs(chg)
            elif chg > 0 : 
                gain += chg 
        avg_lost = lost/period 
        avg_gain = gain/period 
        RSI = (100-100/(1+avg_gain/avg_lost))/100   #scaled RSI values during 0-1
        indicator_RSI.append([RSI])
    indicator_RSI = np.array(indicator_RSI)
    return indicator_RSI 

#########################################################################################
# Training Data (Use as Reference)
#########################################################################################

dataset_import = pd.read_csv('EURUSD.csv')
# init training_set index 
# use dataset from 2000.01.01 to 2015.12.31
start_index = 7084
end_index = 11245 + 1
dataset = dataset_import.iloc[start_index:end_index,:]
training_set = dataset.iloc[:, 2:6].values
look_back_day = 22
sc = MinMaxScaler(feature_range = (0,1))
training_set = sc.fit_transform(training_set)


#########################################################################################
# load prediction model  
#########################################################################################

# load model from json file 
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
print('loaded model')
#eveluate loaded model on test data 
loaded_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#########################################################################################
# Prediction Result 
#########################################################################################
# init input index for test_set 
test_result_day = look_back_day + test_day     
start_input_index = end_index - look_back_day 
end_input_index = start_input_index + test_result_day

inputs = dataset_import.iloc[start_input_index:end_input_index, 2:6].values
inputs = sc.fit_transform(inputs)

X_test = []
for i in range(look_back_day, test_result_day): 
    X_test.append(inputs[i-look_back_day:i, :])
X_test = np.array(X_test)

real_close_price = dataset_import.iloc[end_index:end_input_index, 5:6].values
copy_close_price = real_close_price

predicted_close_price = loaded_model.predict(X_test)
predicted_close_price = sc.inverse_transform(predicted_close_price)
predicted_close_price = predicted_close_price[:, 0]
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
