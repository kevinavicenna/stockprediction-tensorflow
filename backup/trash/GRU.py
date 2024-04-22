import yfinance as yf
from datetime import date
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU,Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping


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
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(1, 1)))
model.add(GRU(50))
model.add(Dense(1))

# Compile and train the GRU model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_data.reshape(-1, 1, 1), train_data.reshape(-1, 1), epochs=10, batch_size=1, verbose=2)

model.summary()
model.save('GRU.keras')


# Kompilasi dan pelatihan model
cp3 = ModelCheckpoint('model2/', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2, callbacks=[cp3, early_stopping], validation_data=(X_test, y_test))

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Misalnya, jika Anda memiliki metrik akurasi:
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Evaluasi model pada data uji
loss = model.evaluate(X_test, y_test)
print("Loss di data uji:", loss)

print("Loss di data uji:", loss)
predict = model.predict(X_test)
# print(predict)

print("------------------------------ Metric -----------------------------------")
print('RMSE : ',sqrt(metrics.mean_squared_error(y_test,predict)))
print('MSE  :  ',metrics.mean_squared_error(y_test,predict))
print('MAE  :  ',metrics.mean_absolute_error(y_test,predict))

print('-------------- Mengecek Apakah Nilainya Overfit atau tidak --------------')
print()


# Simpan model
model.save("lstm_model2.keras")