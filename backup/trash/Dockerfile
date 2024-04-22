FROM continuumio/miniconda3:latest

WORKDIR /app

COPY . /app
COPY requirements.txt app/requirements.txt

RUN apt-get update \
   && pip3 install --upgrade pip \
   && pip3 install --no-cache-dir -r requirements.txt

WORKDIR /app/home-page-stock-prediction

EXPOSE 8501:8501

# menjalankan aplikasi
ENTRYPOINT ["streamlit", "run", "Welcome.py", "--server.port=8501", "--server.address=0.0.0.0"]
