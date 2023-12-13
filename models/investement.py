from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

class StockPricePredictor:
    def __init__(self, stock_symbol, start_date, end_date):
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.model = None
        self.rmse = None

    def download_stock_data(self):
        # Get the stock quote
        self.data = yf.download(self.stock_symbol, start='2012-01-01', end=datetime.now())
        print(self.data.tail(10))
        

    def preprocess_data(self):
        close_data = self.data.filter(['Close'])
        dataset = close_data.values
        training_data_len = int(np.ceil(len(dataset) * 0.95))
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        return scaler, dataset,close_data, scaled_data, training_data_len

    def create_sequences(self, dataset,scaled_data, training_data_len):
        train_data = scaled_data[0:int(training_data_len), :]
        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        test_data = scaled_data[training_data_len - 60: , :]
        
        # Create the data sets x_test and y_test
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
            
        # Convert the data to a numpy array
        x_test = np.array(x_test)

        # Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
        return x_train, y_train, x_test, y_test

    def build_lstm_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train_model(self,scaler, x_train, y_train,x_test, y_test, batch_size=1, epochs=1):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        predictions = self.model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        self.rmse = rmse
        return predictions

    def viz(self,close_data,training_data_len, predictions):
        train = close_data[:training_data_len]
        valid = close_data[training_data_len:]
        valid['Predictions'] = predictions
        print(valid)
        
        plt.figure(figsize=(16, 6))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

        # Adjust the x-axis ticks to show data points for each year
        years = pd.date_range(start=close_data.index[0], end=close_data.index[-1], freq='Y')
        plt.xticks(years, [year.year for year in years])

        plt.show()
            




stock_symbols = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
stock_symbol = 'AAPL'


end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

predictor = StockPricePredictor(stock_symbol, start, end)
predictor.download_stock_data()

scaler, dataset, close_data, scaled_data, training_data_len = predictor.preprocess_data()
x_train, y_train, x_test, y_test = predictor.create_sequences(dataset,scaled_data, training_data_len)
input_shape = x_train.shape[1], 1
predictor.build_lstm_model(input_shape)
predictions = predictor.train_model(scaler, x_train, y_train,x_test, y_test, batch_size=1, epochs=1)
predictor.viz(close_data,training_data_len, predictions)


