import yfinance as yf
from datetime import date
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
from sklearn import metrics
from math import sqrt

# Input nama perusahaan
inputs = input("Masukkan Nama Perusahaan = ")
data = yf.download(inputs, start=date(2018, 1, 1), end=date(2023, 9, 1))
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
print(data)

# Pilih feature
available_features = data.columns[1:]  # Daftar fitur yang tersedia
print("Fitur yang tersedia:", available_features)
input_feats = input("Pilih feature nya = ")

if input_feats not in available_features:
    print("Fitur tidak ditemukan dalam data.")
    exit()

data = data[["Date", input_feats]]
# ...

# Konversi kolom tanggal ke nomor urutan
data["Date"] = pd.to_numeric(data["Date"])

dataset = data.values

train_len = int(np.ceil(len(dataset) * 0.8))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[:train_len, :]

seq_len = 60
x_train, y_train = [], []

for i in range(seq_len, len(train_data)):
    x_train.append(train_data[i - seq_len : i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation="linear"))

model.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"]
)
model.fit(x_train, y_train, batch_size=32, epochs=5)

if train_len == 0:
    train_len = len(scaled_data) - len(train_data)

# Data tes
test_data = scaled_data[train_len - seq_len :, :]
x_test, y_test = [], []

for i in range(seq_len, len(test_data)):
    x_test.append(test_data[i - seq_len : i, 0])
    if train_len + i < len(dataset):
        y_test.append(dataset[train_len + i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predict = model.predict(x_test)
# predictions = scaler.inverse_transform(predictions)

# predict_prices = [price[0] for price in predictions.tolist()]

# print("Prediksi harga:", predict_prices)
print("Prediksi harga:", predict)

print(sqrt(metrics.mean_squared_error(y_test,predict)))
print(metrics.mean_squared_error(y_test,predict))
print(metrics.mean_absolute_error(y_test,predict))