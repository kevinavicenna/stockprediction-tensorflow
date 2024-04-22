import yfinance as yf
from datetime import date
import pandas as pd
import numpy as np
from sklearn import metrics
from math import sqrt
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

inputs = input(str)
data = yf.download(inputs,start=date(2018,1,1),end=date(2023,9,1))
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
# print(data)
# print(data.shape)
# print(len(data))
input_feats= input(str)
data = data[["Date",input_feats]]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[input_feats].values.reshape(-1, 1))


# Create sequences of data for training
sequence_length = 10  # Ganti dengan panjang sekuensial yang sesuai
X, y = [], []
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i+sequence_length])
    y.append(scaled_data[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


X_train = X_train.reshape(X_train.shape[0], 10, 1)
y_train = y_train.reshape(y_train.shape[0], 1)

# Buat model dan lanjutkan seperti yang sebelumnya
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(10, 1)))  # Sesuaikan bentuk input
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=2)



# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Save the model
# model.save("lstm_model.h5")

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Create a new DataFrame with dates and predictions
test_dates = data["Date"].values[train_size+sequence_length:]
lstm_predictions_df = pd.DataFrame({"Date": test_dates, "Predicted_Price": predictions.flatten()})
print(lstm_predictions_df)

























# train_size = int(len(scaled_data) * 0.8)
# test_size = len(scaled_data) - train_size
# train_data, test_data = scaled_data[:train_size, :], scaled_data[train_size:, :]

# model = Sequential()
# # model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
# model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
# model.add(LSTM(50))
# model.add(Dense(1))
# model.summary()

# # Compile and train the LSTM model
# model.compile(loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])
# model.fit(train_data.reshape(-1, 1, 1), train_data.reshape(-1, 1), epochs=10, batch_size=1, verbose=2)

# model.evaluate()

# model.save("lstm.keras")

# # lstm predictions
# lstm_predictions = model.predict(test_data.reshape(-1, 1, 1))
# lstm_predictions = scaler.inverse_transform(lstm_predictions)

# test_dates = pd.to_datetime(data["Date"].values[train_size:])
# # Create a new DataFrame with date and predictions
# lstm_predictions_df = pd.DataFrame({"Date": test_dates, "Predicted_Price": lstm_predictions.flatten()})
# print(test_dates)
# print(lstm_predictions_df)


# # Kompilasi dan pelatihan model
# cp3 = ModelCheckpoint('model2/', save_best_only=True)
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# history = model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=2, callbacks=[cp3, early_stopping], validation_data=(X_test, y_test))


# # Plot loss
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


# # Misalnya, jika Anda memiliki metrik akurasi:
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()





# # Evaluasi model pada data uji
# loss = model.evaluate(X_test, y_test)
# print("Loss di data uji:", loss)

# print("Loss di data uji:", loss)
# predict = model.predict(X_test)
# # print(predict)

# print("------------------------------ Metric -----------------------------------")
# print('RMSE : ',sqrt(metrics.mean_squared_error(y_test,predict)))
# print('MSE  :  ',metrics.mean_squared_error(y_test,predict))
# print('MAE  :  ',metrics.mean_absolute_error(y_test,predict))

# print('-------------- Mengecek Apakah Nilainya Overfit atau tidak --------------')
# print()


# # Simpan model
# model.save("model2.keras")