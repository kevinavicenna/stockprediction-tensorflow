docker stop $(docker ps -q)
docker system prune
docker ps -a