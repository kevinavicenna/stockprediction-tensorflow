docker stop $(docker ps -q)
docker run -p 8501 -d forecast-stock:latest
docker ps -a
