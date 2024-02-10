
import yfinance as yf
from datetime import date
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Input nama perusahaan
stock = 'BBRI.JK'
data = yf.download(stock, start=date(2018, 1, 1), end=date(2023, 9, 1))
# inputs = input("Masukkan Nama Perusahaan = ")
# data = yf.download(inputs, start=date(2018, 1, 1), end=date(2023, 9, 1))
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

# Konversi kolom tanggal ke nomor urutan
data["Date"] = pd.to_numeric(data["Date"])

dataset = data.values

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(dataset)
scaled_data = scaler.fit_transform(dataset)


seq_len = 60

# Memisahkan data menjadi set pelatihan dan set pengujian menggunakan train_test_split
X = []
y = []

for i in range(seq_len, len(scaled_data)):
    X.append(scaled_data[i - seq_len : i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Membangun model LSTM dengan beberapa lapisan tersembunyi
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
# Jumlah unit pada Dense layer sesuaikan dengan jumlah fitur output yang sesuai dengan data
model.add(Dense(2, activation="linear"))

# Kompilasi model
model.compile(optimizer="adam", loss="mean_squared_error")

# Pelatihan model
model.fit(X_train, y_train, batch_size=32, epochs=5)

# Hasil pengujian
predictions = model.predict(X_test)

# Reshape data prediksi menjadi 2 dimensi

# Hasil pengujian
predictions = model.predict(X_test)

# Cek apakah scaler object telah dilatih dengan data yang sama dengan data prediksi
if X_test.shape != predictions.shape:
    # Inversi scaler pada X_test
    X_test_inverse = scaler.inverse_transform(X_test)
    predictions = scaler.inverse_transform(predictions)

# Reshape data prediksi menjadi 1 dimensi
predictions = predictions.reshape((predictions.shape[0],))

print(predictions.shape,y_test.shape)
print(predictions)
print(y_test)

# rmse = np.sqrt(mean_squared_error(y_test, predictions))
# mae = mean_absolute_error(y_test, predictions)

# print("Root Mean Squared Error (RMSE):", rmse)
# print("Mean Absolute Error (MAE):", mae)






