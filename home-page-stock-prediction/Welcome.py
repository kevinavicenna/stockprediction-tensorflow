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
        
    st.header('Fitur Aplikasi My Stock Forecast')
    st.write("""
    Pada aplikasi saya memiliki beberapa opsi pada side bar diantaranya:
    - Halaman Welcome
    - Halaman License pembuat
    - Halaman Prediksi             
             """)

    
if __name__ == '__main__':
    run_program()
