# models/daily_transactions_clustering.py
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class DailyTransactionsClustering:
    def __init__(self, data):
        self.data = data
        self.model = KMeans(n_clusters=4)  # Adjust the number of clusters based on your data


    def preprocess(self):
        # Assuming 'Amount' is a numerical feature
        self.data['Amount'] = StandardScaler().fit_transform(self.data[['Amount']])
        df = pd.read_csv('./Datasets/Transactions.csv')

        self.data['Mode'] = self.data['Mode'].astype('category')
        self.data['Category'] = self.data['Category'].astype('category')

        label_encoder = preprocessing.LabelEncoder()
        self.data['Mode']= label_encoder.fit_transform(self.data['Mode'])
        self.data['Category']= label_encoder.fit_transform(self.data['Category'])
        self.data['Income/Expense']= label_encoder.fit_transform(self.data['Income/Expense'])
        self.data = self.data.drop(columns= ['Date','Subcategory','Note','Currency'])

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.data)
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
        print('Cumulative variance explained by 2 principal components: {:.2%}'.format(np.sum(pca.explained_variance_ratio_)))

        dataset_pca = pd.DataFrame(abs(pca.components_), columns=self.data.columns, index=['PC_1', 'PC_2'])
        print('\n\n', dataset_pca)
        return pca_result

    
    def train(self, pca_result,max_k):
        k = []
        inertias = []
        for i in range(1,max_k+1):
            kmeans = KMeans(n_clusters=i,n_init=i)
            kmeans.fit(self.data)
            
            k.append(i)
            inertias.append(kmeans.inertia_)
            
            
        sns.set_style('whitegrid')
        plt.figure(figsize=(20,8))
        sns.lineplot(x=k,y=inertias,marker='o',dashes=False)


        kmeans = KMeans(n_clusters=4,n_init=10)
        kmeans.fit(pca_result)
        return kmeans

    def cluster_viz(self,kmeans):
        print_data = pd.DataFrame(self.pca_result)
        print_data['clusters'] = kmeans.labels_+1
        centers=pd.DataFrame(kmeans.cluster_centers_)

        plt.figure(figsize=(20,10))
        sns.set_style("whitegrid", {'axes.grid' : False})
        sns.scatterplot(data = print_data,x=print_data[0],y=print_data[1],hue='clusters',legend=False)
        sns.scatterplot(x=centers[0],y=centers[1],c='red',s=200,)

# Example Usage:
# data = ...  # Load your dataset
# transactions_model = DailyTransactionsClustering(data)
# transactions_model.train()