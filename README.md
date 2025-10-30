# Euro-Turkish Lira Exchange Rate Time Series Forecasting Using LSTM 
That project was created using LSTM, which a RNN model. Thus, It might make mistakes.
LSTM was implemented on this project in order to forecast the following 30 days of Turkish Lira-Euro exchange rates. You have to setup some libraries to be able to use that project. Apart from that, do not take serious the results produced using that model. 
The main purpose of the project is to demonstrate how succesfully the LSTM model works on time series data. In addition, the trained model can be utilized on real time seris data as well.
<img width="1818" height="869" alt="Screenshot_19" src="https://github.com/user-attachments/assets/4d9c2af7-4f55-4acc-91be-105c19dea1a1" />
How train, test and validation data were splitted?
Train, test and validation constants weren't splitted randomly or respectively. As our data starts from 0.3 and goes up to 52, we can't split our data in an order. If we appointed 70% of the data as the train constant, train wouldn't see the other part of the data which went up to 52 from 40, which is the highest point, leading a scale problem on the prediction data.
Thus, to prevent such a problem and mistakes in the predictions, we created our train constant splitting our data from different points, allowing the model to consider every condition.

