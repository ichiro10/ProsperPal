# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 07:09:59 2023

@author: erimo
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset (assurez-vous que le fichier CSV est dans le même répertoire que ce script)
df = pd.read_csv('CS_Dataset.csv')

print(df)
# Afficher les premières lignes du dataset
print(df.head())

# Résumé statistique des variables numériques
print(df.describe())

# Vérifier les valeurs manquantes
print(df.isnull().sum())

# Explorer la distribution de l'âge
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Distribution de l\'âge')
plt.show()

# Explorer la répartition des genres
sns.countplot(x='Gender', data=df)
plt.title('Répartition des genres')
plt.show()

# Explorer la relation entre le revenu et l'éducation
sns.scatterplot(x='Income', y='Education', data=df)
plt.title('Revenu par niveau d\'éducation')
plt.show()

# Explorer la corrélation entre les variables numériques
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation')
plt.show()

# Explorer la relation entre le score de crédit et d'autres variables
sns.pairplot(df, hue='Credit Score')
plt.title('Relation entre le score de crédit et d\'autres variables')
plt.show()
