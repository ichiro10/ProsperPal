import streamlit as st 
import pandas as pd 
import numpy as np 
import yfinance as yf 
import plotly.express as px 
import matplotlib.pyplot as plt
import seaborn as sns

#import dataframe
df = pd.read_csv('./Transactions.csv')
#print(df['Date'])

#drop Null values
df.dropna(inplace=True)

#initializing dashboard
st.title("My wallet")
ticker = st.sidebar.text_input('Client ID')



# Use Plotly Express to create the bar chart
fig = px.bar(df, y='Category', x='Amount', title='Expenses By Category', 
             color='Category', labels={'Amount': 'Total Amount'})

# Show the Plotly figure in Streamlit
st.plotly_chart(fig)


# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y %H:%M:%S", errors='coerce', dayfirst=True)

# Get unique years in the dataset
unique_years = sorted(df['Date'].dt.year.unique())
years=[2015, 2016, 2017, 2018 , 'All']

print("unique_years",unique_years)
# Date range filter with default values within the range of your data
start_date_default = pd.to_datetime(df['Date'].min())
end_date_default = pd.to_datetime(df['Date'].max())

# Year filter
selected_year = st.sidebar.selectbox("Select Year", years)

start_date = st.sidebar.date_input('Start Date', value=start_date_default.date(), min_value=df['Date'].min().date(), max_value=df['Date'].max().date())
end_date = st.sidebar.date_input('End Date', value=end_date_default.date(), min_value=df['Date'].min().date(), max_value=df['Date'].max().date())

# Filter data based on year and date range
if selected_year == 'All':
    filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
else : 
   filtered_df = df[(df['Date'].dt.year == selected_year) & (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# Line chart
fig = px.line(filtered_df, x="Date", y="Amount", title="Expense Over Time")
st.plotly_chart(fig)