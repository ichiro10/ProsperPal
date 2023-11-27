# models/investment_regressor.py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class InvestmentRegressor:
    def __init__(self, data):
        self.data = data
        self.model = LinearRegression()

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.data.drop('target', axis=1),  # Assuming 'target' is the label column
            self.data['target'],
            test_size=0.2,
            random_state=42
        )
        self.model.fit(X_train, y_train)

    def evaluate(self):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

# Example Usage:
# data = ...  # Load your dataset
# investment_model = InvestmentRegressor(data)
# investment_model.train()
# investment_model.evaluate()