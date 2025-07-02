# Build the Docker image
docker build -t cctv_service:latest .

# Run the container with GPU support
docker run --gpus all -p 8000:8000 -v /path/on/host/models:/app/data/models -v /path/on/host/temp:/app/data/temp -v /path/on/host/logs:/app/logs efrosine/cctv_service:latest

# Using with Docker Compose
Create a `docker-compose.yml` file with the following content:

```yaml
version: '3'

services:
  cctv_service:
    image: cctv_service:latest
    # Or use the remote image
    # image: efrosine/cctv_service:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/data/models
      - ./temp:/app/data/temp
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

Then run with:

```bash
docker-compose up
```

To run in detached mode (background):

```bash
docker-compose up -d
```

To stop the services:

```bash
docker-compose down
```

