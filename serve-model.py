import os
import json
import requests
import yfinance as yf
import pandas as pd

import datetime
from datetime import date, timedelta

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def run_program():
    start_date = date(2023, 12, 1)
    end_date = date(2024, 3, 31)
    
    input_yf = "BBRI.JK"
    data = yf.download(input_yf, start=start_date, end=end_date)

    # Make date as a column
    data.insert(0, "Date", data.index, True)
    data.reset_index(drop=True, inplace=True)

    # Print data information
    print(f"Date from: {start_date} to: {end_date}")
    print(data)

    columns = "Close"
    data = data[["Date", columns]]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[columns].values.reshape(-1, 1))

    # Split data into train and test sets
    train_size = int(len(scaled_data) * 0.8)
    test_size = len(scaled_data) - train_size
    train_data, test_data = scaled_data[:train_size, :], scaled_data[train_size:, :]


    # Load LSTM model
    url = "http://lstm-server:8501/v1/models/lstm_stock:predict"
    def make_prediction(instances):
        data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
        headers = {"content-type": "application/json"}
        json_response = requests.post(url, data=data, headers=headers)
        predictions = json.loads(json_response.text)
        return predictions

    
    # lstm predictions
    # predictions = make_prediction(test_data.reshape(-1, 1, 1))
    predictions = make_prediction(test_data)
    print(json.dumps(predictions, indent=4))
    # # lstm_predictions = lstm_model.predict(test_data.reshape(-1, 1, 1))
    # # lstm_predictions = lstm_model.predict(test_data)
    # lstm_predictions = scaler.inverse_transform(lstm_predictions)
    
if __name__ == '__main__':
    run_program()
