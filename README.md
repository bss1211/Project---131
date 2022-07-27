
Project-131 -  Crude oil price prediction. 

using :  Artifical Recurrent Neural Network - Long Short Term memory (LSTM)

WTI - West Texas Intermediate
WTI refers to oil extracted from wells in the U.S. and sent via pipeline to Cushing, Oklahoma. The fact that supplies are land-locked is one of the drawbacks to West Texas crude as it’s relatively expensive to ship to certain parts of the globe. The product itself is very light and very sweet, making it ideal for gasoline refining. WTI continues to be the main benchmark for oil consumed in the United States and World.

The dataset used for this project is from - 
https://in.investing.com/

Problem statement : To predict the price of Crude Oil WTI .

In this project we have  used an Artificail recurrent neural network called  Long Short Term Memory(LSTM)

Overview of the project!

Variables used are trading price of each day for past 20 years.

Variables : 
1. Date 
2. Price
3. Open (Price of the Day)
4. High (Price of the Day)
5. Low (Price of the Day)
6. Vol.
7. Change in %

 _ LSTM Model summary _ 

Here we used 3 hidden layers and input shape 100 for first hidden layer

![Model sumary](https://user-images.githubusercontent.com/91157753/181267265-6992da2e-459a-4957-9a20-60edcbc221df.png)


Challenges faced

1.Data collection 
2.Low accuracy with basic models 
3.Model selection 
4.LSTM Model building
5.Implementation of tracer in the deployment 



MODEL DEPLOYEMNET 
Deployement was done using Streamlit Method


![IMG-20220722-WA0000](https://user-images.githubusercontent.com/91157753/181299662-4b61d8c3-76fe-440c-b381-b5211b8da020.jpg)
                                               Web Interface.


![IMG-20220722-WA0003](https://user-images.githubusercontent.com/91157753/181300063-daa6713f-dd7b-4677-a52f-2d2b05cdedec.jpg)

In above plot.

1. Blue line determine the actual values .
2. Orange line determine the predicted value for train data.
3. Green line determine the predicted values for test data. 
4. LSTM is giving better results, so we finalize the LSTM model.


![IMG-20220722-WA0002](https://user-images.githubusercontent.com/91157753/181300090-476d353e-baf0-42a2-9c03-373ced674352.jpg)

1. Blue Line represents past 100 days price.
2. Orange Line represents future 30 days price. 


![IMG-20220722-WA0001](https://user-images.githubusercontent.com/91157753/181299666-01d57ef1-d28d-492e-8afb-cb7b66beaa43.jpg)

WTI – FORECATED PRICE FOR 30 DAYS

Thank you!!
