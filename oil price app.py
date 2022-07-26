import streamlit as st
from datetime import date, timedelta, datetime
import yfinance as yf 
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy
from datetime import date
import yfinance as yf
import pandas_datareader as data
import streamlit as st
from sklearn.preprocessing import power_transform
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import matplotlib.image as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.callbacks import EarlyStopping 

import tensorflow as tf

import math
from sklearn.metrics import mean_squared_error
st.title('Oil Price Prediction')

image = mp.imread(r"C:\Users\sumit\P131-Oil Price Prediction\631987-crudeoil-shutterstock-121117.jpg")
st.image(image)

st.subheader('Crude oil means a mixture of hydrocarbons that exists in liquid phase in natural underground reservoirs and remains liquid at atmospheric pressure after passing through surface separating facilities.')
start = '2010-01-01'
today= date.today().strftime("%Y-%m-%d")

st.sidebar.subheader('Select Date')

start = st.sidebar.date_input("Start date", datetime.date(2003,6,2))

end = st.sidebar.date_input("End Date",datetime.date(2022, 7, 1))

df = pd.read_csv(r"C:\Users\sumit\P131-Oil Price Prediction\Crude Oil WTI Futures Historical Data.csv")



st.subheader('Latest Price')
st.write(df.tail(1))




check_box = st.sidebar.checkbox(label="Dispaly Historical Data")

if check_box:
    st.subheader('Historical Data')
    st.write(df)
    


#n_years = st.sidebar.slider("YEARS DATA",1,10)
#period = n_years*365

#st.subheader('Historical Data')
#st.write(df)




def plot_graph():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.Date,y=df['Price'], name="Price", line_color='blue'))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    return fig

plot_graph()

train_df = df.sort_values(by=['Date']).copy()

df1 = df.reset_index()['Price']

scaler=MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.80)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model = Sequential()

# Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables
n_neurons = X_test.shape[1] * X_train.shape[2]
print(n_neurons, X_train.shape[1], X_train.shape[2])
model.add(LSTM(n_neurons, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))) 
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Training the model
epochs = 3
batch_size = 64
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,validation_data=(X_test, ytest))


#model = load_model('Staked_LSTM.h5')

## Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

# plot baseline and predictions
st.title('Actual vs Prediction')
fig = plt.figure(figsize=(10,5))
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.legend(["Test",'Train',"Prediction"], loc="upper left")
st.pyplot(fig)
user_input1 = len(test_data)-100
x_input=test_data[user_input1:].reshape(1,-1)   

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# demonstrate prediction for next days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<31):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1,(len(temp_input)+100)+-n_steps-1, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        #print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        #print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

#st.write(lst_output)

df4 = scaler.inverse_transform((lst_output))

day_new=np.arange(1,101)               # As we know we taking 100 as time stamp, that's we considering (1,101)
day_pred=np.arange(101,132)            # As we planning to preditic 30days,(101+31=131)

input_data2 = (len(df1)-100)

st.title ('Oil Forecast')

st.subheader('Forecast Graph')
fig3=plt.figure(figsize=(10,5))
plt.plot(day_new,scaler.inverse_transform(df1[input_data2:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
st.pyplot(fig3)


df3=df1.tolist()
df3.extend(lst_output)

df3 = scaler.inverse_transform(df3).tolist()

forecast_data = pd.date_range(start="2022-07-01",end="2022-07-31",freq='D',name='Date')
forecast_data = pd.DataFrame(data=forecast_data)
forecast_data['Price_new'] = scaler.inverse_transform(lst_output)
df = df[['Date','Price']]
df['Date'] = pd.to_datetime(df['Date'])
full_data = pd.concat([df,forecast_data])
st.subheader('Forecasted Values')
st.write(forecast_data)

def plot_graph():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_data.Date,y=forecast_data['Price_new'], name="Price", line_color='blue'))
    fig.layout.update(title_text='Forecast data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    return fig

plot_graph()


# Range for 1 month
st.title('Prediction Range for next 30Days in dollars')
st.subheader(forecast_data.Price_new.min())
st.subheader('-')

st.subheader(forecast_data.Price_new.max())



