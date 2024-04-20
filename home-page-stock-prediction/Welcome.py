import os
import json
import requests
import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def run_program():
    # Judul halaman
    st.set_page_config(
        page_title="My Stock Forecast",
        page_icon="🔫"
    )
    
    st.markdown("""<h1 style='text-align: center;'>My Stock Forecast</h1>""", unsafe_allow_html=True)
    st.image("https://th.bing.com/th/id/OIG3.6QjRqg5YProOXqaHU0nW?w=1024&h=1024&rs=1&pid=ImgDetMain")
        
    st.write("""
    ## Arsitecture LSTM (Long Short Term Memory)
    Konsep dari jaringan saraf LSTM (Long Short-Term Memory) ialah sejenis desain jaringan saraf tiruan yang memodelkan dan memproses rangkaian data termasuk teks, suara, dan deret waktu.
    Dibandingkan dengan jaringan saraf rekursif (RNN) yang lebih sederhana, LSTM memiliki fitur sel memori yang memungkinkannya menyimpan informasi jangka panjang, menghindari masalah gradien hilang, dan mengontrol aliran informasi melalui gerbang lupa, gerbang masukan, dan gerbang keluaran.
    
    Hal ini membuat LSTM berguna dalam berbagai aplikasi, termasuk pemrosesan bahasa alami, pengenalan suara, terjemahan mesin, dan pemodelan deret waktu.
    Dengan teknik ini, kecerdasan buatan dapat belajar dari pola masa lalu untuk memprediksi pergerakan harga masa depan.
    """)

    st.write("""
    ## Bagaimana LSTM Bekerja pada Prediksi Saham?
    LSTM memiliki kemampuan untuk mempertahankan dan menggunakan informasi jangka panjang dari data historis.
    Dalam konteks prediksi saham, LSTM dapat mempelajari hubungan kompleks antara variabel-variabel seperti
    harga saham sebelumnya, volume perdagangan, dan faktor-faktor lainnya.

    Namun,LSTM melakukan ini dengan mempertimbangkan informasi sebelumnya dalam jangka waktu tertentu, dan kemudian
    memutuskan bagaimana informasi tersebut akan mempengaruhi pergerakan harga saham pada waktu selanjutnya.
    """)
        
    st.header('Fitur aplikasi My Stock Forecast')
    st.write("""
    Pada aplikasi saya memiliki beberapa opsi pada side bar diantaranya:
    - Halaman Welcome
    - Halaman License pembuat
    - Halaman Prediksi             
             """)

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

    # Data visualization
    st.header("Data visualization")

    # Plot the data using Plotly
    fig = px.line(data, x="Date", y=data.columns, title="price of the stock", width=1000, height=800)
    st.plotly_chart(fig)

    # Select the column to be used in forecasting
    columns = st.selectbox('Select the column to be used in forecasting', data.columns[1:])
    data = data[["Date", columns]]
    st.write("Selected data")
    st.write(data)

    # Decompose the data
    st.header('Decomposition')
    st.text('Seasonal Decompose adalah sebuah fungsi yang digunakan sebagai moving average')
    decompose = seasonal_decompose(data[columns], model='additive', period=12)
    st.plotly_chart(px.line(x=data["Date"], y=decompose.trend, title='Trend', labels={"x": "Date", "y": "Price"}).update_traces(line_color="Red"))
    st.plotly_chart(px.line(x=data["Date"], y=decompose.seasonal, title='Seasonality', labels={"x": "Date", "y": "Price"}).update_traces(line_color="Green"))
    st.plotly_chart(px.line(x=data["Date"], y=decompose.resid, title='Residual', labels={"x": "Date", "y": "Price"}).update_traces(line_dash="dot"))


    st.header('Time Series LSTM Model')
    st.write('**Note**: LSTM Model is trained and predicted on the selected column of the data. Make sure using data correct before using')

    # Prepare the data for LSTM model
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
    st.write("Data Stock",data)
    st.write("Data Json",data_json)
    response = requests.post(url, data=data_json, headers=headers)
    st.write(response)

    lstm_predictions = response.json()
    st.write("lstm_predictions",lstm_predictions)
    
    predictions_list = lstm_predictions["predictions"]

    # Convert the nested list into a flat list of floats
    flat_predictions = [item for sublist in predictions_list for item in sublist]

    # Convert the flat list to a numpy array
    lstm_predictions_array = np.array(flat_predictions)

    # Reshape the array to a column vector
    lstm_predictions_array = lstm_predictions_array.reshape(-1, 1)

    # Apply inverse transformation
    lstm_predictions = scaler.inverse_transform(lstm_predictions_array)
    st.write(lstm_predictions)

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
    
if __name__ == '__main__':
    run_program()
