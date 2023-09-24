import yfinance as yf
from datetime import date
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense


inputs = input(str)
data = yf.download(inputs,start=date(2018,1,1),end=date(2023,9,1))
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
print(data)

input_feats= input(str)
data = data[["Date",input_feats]]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[input_feats].values.reshape(-1, 1))

train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[:train_size, :], scaled_data[train_size:, :]

lstm_model = Sequential()
# lstm_model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))

# Compile and train the LSTM model
lstm_model.compile(loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])
lstm_model.fit(train_data.reshape(-1, 1, 1), train_data.reshape(-1, 1), epochs=10, batch_size=1, verbose=2)
lstm_model.summary()

# lstm_model.evaluate()

# lstm_model.save("lstm.keras")

# # lstm predictions
# lstm_predictions = lstm_model.predict(test_data.reshape(-1, 1, 1))
# lstm_predictions = scaler.inverse_transform(lstm_predictions)

# test_dates = pd.to_datetime(data["Date"].values[train_size:])
# # Create a new DataFrame with date and predictions
# lstm_predictions_df = pd.DataFrame({"Date": test_dates, "Predicted_Price": lstm_predictions.flatten()})
# print(test_dates)
# print(lstm_predictions_df)