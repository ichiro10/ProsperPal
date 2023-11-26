# models/daily_transactions_clustering.py
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class DailyTransactionsClustering:
    def __init__(self, data):
        self.data = data
        self.model = KMeans(n_clusters=3)  # Adjust the number of clusters based on your data

    def preprocess(self):
        # Assuming 'Amount' is a numerical feature
        self.data['Amount'] = StandardScaler().fit_transform(self.data[['Amount']])

    def train(self):
        self.preprocess()
        self.model.fit(self.data[['Amount']])

    def predict(self, new_data):
        new_data['Amount'] = StandardScaler().fit_transform(new_data[['Amount']])
        return self.model.predict(new_data[['Amount']])

# Example Usage:
# data = ...  # Load your dataset
# transactions_model = DailyTransactionsClustering(data)
# transactions_model.train()
# new_data = ...  # New data for prediction
# predictions = transactions_model.predict(new_data)