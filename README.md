# Euro-Turkish Lira Exchange Rate Time Series Forecasting Using LSTM 

That project was created using LSTM, which a RNN model. It might make mistakes, thus do not take the results serious.

LSTM was implemented on this project in order to forecast Turkish Lira-Euro exchange rates for the following 30 days. You have to setup some libraries to be able to progress. Apart from that, do not take serious the results produced using that model. 
The main purpose of the project is to demonstrate how succesfully the LSTM model works on the time series data. In addition, the trained model can be utilized on real time seris data as well. As seen below, predictions made by using the test data after the training stage looks well-overlapped. 
<img width="1000" height="500" alt="TheResultofAWindow" src="https://github.com/user-attachments/assets/9d126460-9230-4d70-9bae-5795b4464fcb" />
However, And below, using the last window of the data, the exchange rate in the following 30 days was predicted. Since the recursive forecasting implemented on the last window, predictions would more likely be wrong. 
<img width="1000" height="500" alt="ThePrediction" src="https://github.com/user-attachments/assets/08d9440b-ba7a-4283-b8ce-001f2c21e6a4" />
