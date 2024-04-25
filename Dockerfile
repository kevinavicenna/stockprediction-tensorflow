FROM python:3.11.9-slim-bookworm

WORKDIR /app

COPY . /app
COPY requirements.txt app/requirements.txt

RUN apt-get update && apt-get install -y \
   python3-pip \
   && pip3 install --no-cache-dir -r requirements.txt

WORKDIR /app/home-page-stock-prediction

EXPOSE 8502:8502

ENTRYPOINT ["streamlit", "run", "Welcome.py", "--server.port=8502", "--server.address=0.0.0.0"]
