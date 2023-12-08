# models/investment_regressor.py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
from alpha_vantage.timeseries import TimeSeries

label = "Description"

class InvestmentRegressor:
    def __init__(self, data):
        self.data = data
        self.model = LinearRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self):
        # Check if label column is present
        if label not in self.data.columns:
            print(f"Error: '{label}' not found in the dataset.")
            return

        # Assuming 'label' is the column with dates
        y = self.data[label]

        # Drop rows with invalid dates
        self.data = self.data[pd.to_datetime(y, errors='coerce').notna()]

        # Extract features from dates
        self.data['Year'] = pd.to_datetime(self.data[label]).dt.year
        self.data['Month'] = pd.to_datetime(self.data[label]).dt.month
        self.data['Day'] = pd.to_datetime(self.data[label]).dt.day
        self.data['Weekday'] = pd.to_datetime(self.data[label]).dt.weekday

        # Check if all necessary columns are present
        if self.data.empty or 'Year' not in self.data.columns or 'Month' not in self.data.columns or 'Day' not in self.data.columns or 'Weekday' not in self.data.columns:
            print("Error: Columns missing after data cleaning.")
            return

        # Split the data into training and testing sets
        X = self.data.drop(label, axis=1)
        y = self.data[label]
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
chemin_du_fichier = 'personal-investments.csv'

# Lire le fichier CSV
data = pd.read_csv(chemin_du_fichier, parse_dates=[label])  # Load your dataset

investment_model = InvestmentRegressor(data)
investment_model.train()
investment_model.evaluate()


