import streamlit as st 
import pandas as pd 
import numpy as np 
import yfinance as yf 
import plotly.express as px 
import matplotlib.pyplot as plt
import seaborn as sns

#import sketch

df = pd.read_csv('./Transactions.csv')

df.dropna(inplace=True)

st.title("My wallet")
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start Date')
end_date =  st.sidebar.date_input('End Date')

#plt.title('Expence By Category')
# Use Plotly Express to create the bar chart
fig = px.bar(df, y='Category', x='Amount', title='Expenses By Category', 
             color='Category', labels={'Amount': 'Total Amount'})

# Show the Plotly figure in Streamlit
st.plotly_chart(fig)