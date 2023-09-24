import yfinance as yf
from datetime import date
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU,Dense


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

# Create the GRU model
gru_model = Sequential()
gru_model.add(GRU(50, return_sequences=True, input_shape=(1, 1)))
gru_model.add(GRU(50))
gru_model.add(Dense(1))

# Compile and train the GRU model
gru_model.compile(loss='mean_squared_error', optimizer='adam')
gru_model.fit(train_data.reshape(-1, 1, 1), train_data.reshape(-1, 1), epochs=10, batch_size=1, verbose=2)

gru_model.summary()
gru_model.save('GRU.keras')
