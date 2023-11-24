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



def perform(y_pred,y_test,clf_name):
    print("Precision : ", precision_score(y_test, y_pred,average='micro'))
    print("Recall : ", recall_score(y_test, y_pred,average='micro'))
    print("Accuracy : ", accuracy_score(y_test, y_pred))
    print("F1 Score : ", f1_score(y_test, y_pred,average='micro'))
    print('')
    print(confusion_matrix(y_test, y_pred), '\n')
    cm = confusion_matrix(y_test, y_pred)
    # Créer une heatmap pour visualiser la matrice de confusion
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['low', 'average', 'high'],
                yticklabels=['low', 'average', 'high'])
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies étiquettes')
    plt.title('Matrice de Confusion :'+clf_name)
    plt.show()





def data_preprocessing(data):
        print(data)
        print(data.info())
        print(data.nunique())
        print(data.describe(include='all'))

        #missed values 
        print(data.isnull().sum())

        #categorical data 
        categorical_features = ['Gender','Education','Marital Status','Home Ownership'] 

        for feature in categorical_features:
            # Label Encoding
            label_encoder = LabelEncoder()
            data[feature+'_encoded'] = label_encoder.fit_transform(data[feature])
            data = data.drop(columns=[feature])
        print(data)
        print(data.info())
        print(data.describe(include='all'))
        

def data_viz(df):
      
        sns.pairplot(df)
        plt.show()
        

        # Boxplot for each numerical variable
        df.boxplot(figsize=(12, 6))
        plt.show()

        # Countplot for categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            sns.countplot(x=col, data=df)
            plt.show()
                
        p = sns.histplot(df["Number of Children"], color="red", kde=True, bins=50, alpha=1, fill=True, edgecolor="black")
        p.axes.lines[0].set_color("#101B15")
        p.axes.set_title("\n Number of Children Distribution\n", fontsize=25)
        plt.ylabel("Count", fontsize=20)
        plt.xlabel("Number of Children", fontsize=20)
        sns.despine(left=True, bottom=True)
        plt.show()

        
        p = sns.histplot(df["Age"], color="green", kde=True, bins=50, alpha=1, fill=True, edgecolor="black")
        p.axes.lines[0].set_color("#101B15")
        p.axes.set_title("\n Age Distribution\n", fontsize=25)
        plt.ylabel("Count", fontsize=20)
        plt.xlabel("Age", fontsize=20)
        sns.despine(left=True, bottom=True)
        plt.show()




def run(csv: str = './CS_Dataset.csv'):
        data = pd.read_csv(csv)
        print(data)
        print(data.info())

        X = data.drop('Credit Score', axis=1)
        y = data['Credit Score']
  

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if True :
            # Define numeric and categorical features
            numeric_features = X.select_dtypes(include=['int64']).columns
            categorical_features = X.select_dtypes(include=['object']).columns

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

            # Create a pipeline with the preprocessor and a classifier (Random Forest in this case)
            classifier = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier())
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
                # Define the pipeline
                classifier.set_params(classifier=clf)
                
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
                if 'best_model' not in locals() or avg_accuracy > best_model['accuracy']:
                    best_model = {'name': clf_name, 'model': classifier, 'accuracy': avg_accuracy}

            # Save the best model to a file
            if 'best_model' in locals():
                model_filename = 'best_model.joblib'
                joblib.dump(best_model['model'], model_filename)
                print(f"\nBest model '{best_model['name']}' saved to {model_filename}")

            

if __name__ == '__main__':
    if len(sys.argv) > 1:
        run(folder=sys.argv[1])
    else:
        run()        