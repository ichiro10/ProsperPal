# models/credit_score_classifier.py
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_curve, auc
import joblib


class CreditScoreClassifier:
    def __init__(self, data):
        self.data = data
        self.best_model = None


    def train(self):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop('Credit Score', axis=1), self.data['Credit Score'], test_size=0.2, random_state=42)
        
        # Define numeric and categorical features
        numeric_features = X_train.select_dtypes(include=['int64']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns

        # Create transformers for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder())
        ])

        # Combine transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # List of classifiers
        clfs = [
            ('Logistic Regression', LogisticRegression()),
            ('SVM', SVC()),
            ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=3)),
            ('Random Forest', RandomForestClassifier()),
            ('DecisionTree', DecisionTreeClassifier()),
            ('GradientBoosting', GradientBoostingClassifier())
        ]

        for clf_name, clf in clfs:
            # Create a pipeline with the preprocessor and the current classifier
            classifier = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', clf)
            ])

            # Fit the pipeline on the training data
            classifier.fit(X_train, y_train)

            # Cross-validate the model
            scores = cross_validate(classifier, X_train, y_train, scoring="accuracy", cv=5)
            avg_accuracy = scores['test_score'].mean()

            # Print evaluation metrics
            print(clf_name)
            print("Average Accuracy:", avg_accuracy)

            y_pred = classifier.predict(X_test)
            print("Precision:", precision_score(y_test, y_pred, average='micro'))
            print("Recall:", recall_score(y_test, y_pred, average='micro'))
            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("F1 Score:", f1_score(y_test, y_pred, average='micro'))

            # Save the best model
            if self.best_model is None or avg_accuracy > self.best_model['accuracy']:
                self.best_model = {'name': clf_name, 'model': classifier, 'accuracy': avg_accuracy}

        # Save the best model to a file
        if self.best_model is not None:
            model_filename = 'best_model.joblib'
            joblib.dump(self.best_model['model'], model_filename)
            print(f"\nBest model '{self.best_model['name']}' saved to {model_filename}")

# Example usage
# classifier_instance = CreditScoreClassifier(your_data)
# classifier_instance.train()