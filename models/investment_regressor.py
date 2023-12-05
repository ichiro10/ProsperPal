# models/investment_regressor.py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
from alpha_vantage.timeseries import TimeSeries

# Remplacez 'YOUR_API_KEY' par votre clé d'API Alpha Vantage
api_key = 'YOUR_API_KEY'
symbol = 'MSFT'
interval = 'daily'  # Peut être 'daily', 'weekly', 'monthly', etc.
start_date = '2022-01-01'
end_date = '2023-01-01'

# Initialiser la classe TimeSeries avec votre clé d'API
ts = TimeSeries(key=api_key, output_format='pandas')

# Obtenir les données du cours des actions
data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')

# Filtrer les données pour la période spécifiée
filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]

# Afficher les premières lignes des données
print(filtered_data.head())

# Enregistrer les données dans un fichier CSV
filtered_data.to_csv('stock_data_alpha_vantage.csv')

class InvestmentRegressor:
    def __init__(self, data):
        self.data = data
        self.model = LinearRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self):
        # Assuming 'target' is the label column
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        # Ensure that the train method has been called before evaluating
        if self.X_test is None or self.y_test is None:
            print("Error: Model not trained. Call the 'train' method first.")
            return

        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)
        
        # Evaluate the model
        mse = mean_squared_error(self.y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

# Example Usage:
# data = ...  # Load your dataset
# investment_model = InvestmentRegressor(data)
# investment_model.train()
# investment_model.evaluate()
