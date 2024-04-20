# Menggunakan image tensorflow
FROM tensorflow/tensorflow:latest

# Tambahkan direktori di dalam container
WORKDIR /app

# Salin semua file ke dalam container
COPY . /app
COPY requirements.txt app/requirements.txt

# Install dependencies yang dibutuhkan
# RUN apt-get update && apt-get install -y \
#      python3-pip \
#      && pip3 install --upgrade pip \
#      && pip3 install streamlit yfinance pandas numpy matplotlib \
#      seaborn plotly statsmodels scikit-learn plotly_express

RUN apt-get update && apt-get install -y \
   python3-pip \
   && pip3 install --upgrade pip \
   && pip3 install --no-cache-dir -r requirements.txt

WORKDIR /app/home-page-stock-prediction

# Port yang digunakan
EXPOSE 8501:8501

# Command menjalankan aplikasi
ENTRYPOINT ["streamlit", "run", "Welcome.py", "--server.port=8501", "--server.address=0.0.0.0"]
