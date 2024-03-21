# Gunakan base image TensorFlow yang telah disediakan oleh Docker Hub
FROM tensorflow/tensorflow:latest

# Install dependencies yang dibutuhkan :D
RUN apt-get update && apt-get install -y \
    python3-pip \
    && pip3 install --upgrade pip \
    && pip3 install streamlit yfinance pandas numpy matplotlib \
    seaborn plotly statsmodels scikit-learn plotly_express

# Tambahkan direktori kerja di dalam container
WORKDIR /app

# Salin semua file yang diperlukan ke dalam container
COPY . /app

# Port yang digunakan oleh Streamlit
EXPOSE 8501:8501

# Perintah untuk menjalankan aplikasi Streamlit
ENTRYPOINT ["streamlit", "run", "home-page-stock-prediction/Welcome.py", "--server.port=8501", "--server.address=0.0.0.0"]