#!/usr/bin/env python
# coding: utf-8

# In[62]:


import warnings 
warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Define function to load and preprocess data
@st.cache
def load_data():
    # Load data from yfinance
    df = yf.download("RELIANCE.NS", start="2015-01-01", end="2022-12-30")
    # Reset index to move date from upper header to main column
    df = df.reset_index()
    # Filter columns to only include date and close price
    df = df[['Date', 'Close']]
    # Rename columns
    df.columns = ['date', 'close']
    # Add features for year, month, day of week, and day of month
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    # Convert date to string format
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df

# Load data
df = load_data()

# Split data into training and testing sets
train = df[df['date'] < '2022-06-01']
test = df[df['date'] >= '2022-06-01']

# Define X and y for training and testing sets
X_train = train.drop(['date', 'close'], axis=1)
y_train = train['close']
X_test = test.drop(['date', 'close'], axis=1)
y_test = test['close']

# Train random forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions for next 30 days starting from 2023-01-01
future_dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]
future_features = pd.DataFrame({'date': future_dates_str})
future_features['year'] = future_features['date'].str.slice(0, 4).astype(int)
future_features['month'] = future_features['date'].str.slice(5, 7).astype(int)
future_features['day_of_week'] = pd.to_datetime(future_features['date']).dt.dayofweek
future_features['day_of_month'] = pd.to_datetime(future_features['date']).dt.day
future_features = future_features.drop('date', axis=1)
future_predictions = rf.predict(future_features)

# Create table of future dates and predicted close prices
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions})
future_df['Date'] = future_df['Date'].dt.strftime('%Y-%m-%d')
future_df = future_df.set_index('Date')
st.table(future_df)


import matplotlib.pyplot as plt

# plot the line graph
plt.plot(future_predictions)

# add title and axis labels
plt.title("Predictions for Next 30 Days")
plt.xlabel("Days")
plt.ylabel("predictions")

# rotate x-axis labels for better readability
plt.xticks(rotation=45)
st.set_option('deprecation.showPyplotGlobalUse', False)
# display the plot
st.pyplot()


# In[ ]:





# In[ ]:




