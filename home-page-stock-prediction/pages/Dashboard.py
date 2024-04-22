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


st.title("Stock Forecast for Market Indonesia")

# Sidebar
st.sidebar.title("Select the Parameters Below")
start_date = st.sidebar.date_input("Start Date", date(2023,12,1))
end_date = st.sidebar.date_input("End Date", date(2024, 3,31))

# Ticker selection
input_yf_list = ["BBRI.JK","GOTO.JK","ADRO.JK"]
input_yf = st.sidebar.selectbox("Select the Ticker", input_yf_list)

# Fetch Data from Yahoo Finance
data = yf.download(input_yf, start=start_date, end=end_date)

# Make date as a column
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write("Date from", start_date, "to", end_date)
st.write(data)

st.write("---")
st.header("Stock Price Visualization")

# Plot the data using Plotly
fig = px.line(data, x="Date", y=data.columns, width=1000, height=800)
st.plotly_chart(fig)

# Select the column to be used in forecasting
columns = st.selectbox('Select the column to be used in forecasting', data.columns[1:])
data = data[["Date", columns]]
st.write("Selected data")
st.write(data)

st.write("---")
st.header('Decomposition')
st.write('Seasonal Decompose is a function that is used as a moving average')
decompose = seasonal_decompose(data[columns], model='additive', period=12)
st.plotly_chart(px.line(x=data["Date"], y=decompose.trend, title='Trend', labels={"x": "Date", "y": "Price"}).update_traces(line_color="Red"))
st.plotly_chart(px.line(x=data["Date"], y=decompose.seasonal, title='Seasonality', labels={"x": "Date", "y": "Price"}).update_traces(line_color="Green"))
st.plotly_chart(px.line(x=data["Date"], y=decompose.resid, title='Residual', labels={"x": "Date", "y": "Price"}).update_traces(line_dash="dot"))

st.write("---")
st.header('Time Series LSTM Model')
st.write('**Note**: LSTM Model is trained and predicted on the selected column of the data. Make sure using correct column to predict')

# scaling model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[columns].values.reshape(-1, 1))

# Split data into train and test sets
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[:train_size, :], scaled_data[train_size:, :]


# Load LSTM model
# url = "http://localhost:8501/v1/models/lstm_stock/metadata"
url = "http://localhost:8501/v1/models/lstm_stock:predict"
# url = "http://lstm-server:8501/v1/models/lstm_stock:predict"

response = requests.get(url)
headers = {"content-type": "application/json"}
signature_name = "serving_default"

# Consider the first 10 data_test images 
instances = test_data.reshape(-1, 1, 1).tolist()

# Create a dictionary
data_dict = {
    "signature_name": signature_name,
    "instances": instances
}
data_json = json.dumps(data_dict)
st.write("Data Json",data_json)

st.write("Data Stock",data)

response = requests.post(url, data=data_json, headers=headers)

lstm_predictions = response.json()
# st.write("lstm_predictions",lstm_predictions) # JSON

predictions_list = lstm_predictions["predictions"]
# Convert the nested list into a flat list of floats
flat_predictions = [item for sublist in predictions_list for item in sublist]
# Convert the flat list to a numpy array
lstm_predictions_array = np.array(flat_predictions)
# Reshape the array to a column vector
lstm_predictions_array = lstm_predictions_array.reshape(-1, 1)
# Apply inverse transformation
lstm_predictions = scaler.inverse_transform(lstm_predictions_array)
# st.write(lstm_predictions) #Convert from json
test_dates = pd.to_datetime(data["Date"].values[train_size:])

# Create a new DataFrame with date and predictions
lstm_predictions_df = pd.DataFrame({"Date": test_dates, "Predicted_Price": lstm_predictions.flatten()})

# Display the LSTM predictions
st.write("## Result LSTM Prediction")
st.write(lstm_predictions_df)
st.write("---")

# Plot the LSTM predictions
fig_lstm = go.Figure()
fig_lstm.add_trace(go.Scatter(x=data["Date"], y=data[columns], name="Actual Data"))
fig_lstm.add_trace(go.Scatter(x=lstm_predictions_df["Date"], y=lstm_predictions_df["Predicted_Price"], name="LSTM Predictions", mode="lines", line=dict(color="Red")))
fig_lstm.update_layout(title_text="Vizualization Actual Data vs LSTM Predictions", xaxis_title="Date", yaxis_title="Price", width=1000, height=400)
st.plotly_chart(fig_lstm)