# Svema Services

Microservices for the Svema photo management application.

## Services

### face-recognition
Python-based face detection and recognition service using dlib.
- Face detection with HOG model
- 128-dimensional face encodings
- DBSCAN clustering support
- REST API on port 5555

### shot-captioning
AI-powered image captioning service using BLIP (Bootstrapping Language-Image Pre-training).
- Automatic generation of descriptive captions for photos
- Batch processing of uncaptioned shots
- Direct database integration
- REST API on port 5556

## Deployment

All services can be deployed together using docker-compose:

```bash
docker compose up -d --build
```

Or deploy individual services:

```bash
docker compose up -d face-recognition
```

## Network

All services connect to the `svema-network` bridge network to communicate with the main Svema application.

## Development

Each service has its own directory with:
- Dockerfile
- requirements.txt or package dependencies
- Source code
- README with service-specific documentation
