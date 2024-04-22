import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


st.set_page_config(
    page_title="My Stock Forecast",
    page_icon="🔫"
    )

st.title("Stock Forecast")
# st.sidebar.image("https://th.bing.com/th/id/OIG3.6QjRqg5YProOXqaHU0nW?w=1024&h=1024&rs=1&pid=ImgDetMain")

# Sidebar
# st.sidebar.title("Select the Parameters Below")
# start_date = st.sidebar.date_input("Start Date", date(2023,12,1))
# end_date = st.sidebar.date_input("End Date", date(2024, 1,31))

# # Ticker selection
# input_yf_list = ["BBRI.JK","GOTO.JK","ADRO.JK"]
# input_yf = st.sidebar.selectbox("Select the Ticker", input_yf_list)

# # Fetch Data from Yahoo Finance
# data = yf.download(input_yf, start=start_date, end=end_date)

# # Make date as a column
# data.insert(0, "Date", data.index, True)
# data.reset_index(drop=True, inplace=True)
# st.write("Date from", start_date, "to", end_date)
# st.write(data)

# # Data visualization
# st.header("Data visualization")

# # Plot the data using Plotly
# fig = px.line(data, x="Date", y=data.columns, title="price of the stock", width=1000, height=800)
# st.plotly_chart(fig)

# # Select the column to be used in forecasting
# columns = st.selectbox('Select the column to be used in forecasting', data.columns[1:])
# data = data[["Date", columns]]
# st.write("Selected data")
# st.write(data)
    
# # Decompose the data
# st.header('Decomposition')
# st.text('Seasonal Decompose adalah sebuah fungsi yang digunakan sebagai moving average')
# decompose = seasonal_decompose(data[columns], model='additive', period=12)
# # st.write(decompose.plot())

# st.plotly_chart(px.line(x=data["Date"], y=decompose.trend, title='Trend', labels={"x": "Date", "y": "Price"}).update_traces(line_color="Red"))
# st.plotly_chart(px.line(x=data["Date"], y=decompose.seasonal, title='Seasonality', labels={"x": "Date", "y": "Price"}).update_traces(line_color="Green"))
# st.plotly_chart(px.line(x=data["Date"], y=decompose.resid, title='Residual', labels={"x": "Date", "y": "Price"}).update_traces(line_dash="dot"))


# # LSTM Model
# st.header('LSTM Model')
# st.write('**Note**: LSTM Model is trained and predicted on the selected column of the data. Make sure using data correct before using')

# # Prepare the data for LSTM model
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data[columns].values.reshape(-1, 1))

# # Split data into train and test sets
# train_size = int(len(scaled_data) * 0.8)
# test_size = len(scaled_data) - train_size
# train_data, test_data = scaled_data[:train_size, :], scaled_data[train_size:, :]


# # Create the LSTM model
# lstm_model = tf.keras.models.load_model('lstm_model2.keras')
# # lstm_model = tf.keras.models.load_model('lstm_model.keras')

# # lstm predictions
# lstm_predictions = lstm_model.predict(test_data.reshape(-1, 1, 1))
# lstm_predictions = scaler.inverse_transform(lstm_predictions)

# # Convert the index to datetime
# test_dates = pd.to_datetime(data["Date"].values[train_size:])

# # Create a new DataFrame with date and predictions
# lstm_predictions_df = pd.DataFrame({"Date": test_dates, "Predicted_Price": lstm_predictions.flatten()})

# # Display the LSTM predictions
# st.write("## LSTM Predictions")
# st.write(lstm_predictions_df)
# st.write("---")

# # Plot the LSTM predictions
# fig_lstm = go.Figure()
# fig_lstm.add_trace(go.Scatter(x=data["Date"], y=data[columns], name="Actual Data"))
# fig_lstm.add_trace(go.Scatter(x=lstm_predictions_df["Date"], y=lstm_predictions_df["Predicted_Price"], name="LSTM Predictions", mode="lines", line=dict(color="Red")))
# fig_lstm.update_layout(title_text="Actual Data vs LSTM Predictions", xaxis_title="Date", yaxis_title="Price", width=1000, height=400)
# st.plotly_chart(fig_lstm)