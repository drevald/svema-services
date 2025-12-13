# Shot Captioning Service

AI-powered image captioning service for Svema photos using BLIP (Bootstrapping Language-Image Pre-training).

## Features

- **Automatic Image Captioning**: Generates descriptive captions for photos using BLIP model
- **Database Integration**: Directly updates shot comments in Svema database
- **Batch Processing**: Harvest and process multiple shots at once
- **REST API**: Simple HTTP endpoints for integration

## API Endpoints

### Health Check
```bash
GET /health
```
Returns service status and model information.

### Caption Single Image
```bash
POST /caption
Content-Type: multipart/form-data

image: <image file>
```
Generates caption for uploaded image.

### Caption Specific Shot
```bash
POST /caption/shot/{shot_id}
```
Generates caption for shot by ID and updates database.

### Harvest Uncaptioned Shots
```bash
POST /harvest
Content-Type: application/json

{
  "limit": 100,        # Optional: max shots to process (default: 100)
  "album_id": 123      # Optional: only process specific album
}
```
Processes all shots without captions and updates database.

## Model

Uses **Salesforce/blip-image-captioning-base** which generates descriptive captions like:
- "a woman sitting on a bench in a park"
- "a sunset over the ocean with boats"
- "a group of people standing in front of a building"

## Configuration

Environment variables:
- `DB_HOST`: Database host (default: svema-postgres-1)
- `DB_PORT`: Database port (default: 5433)
- `DB_NAME`: Database name (default: svema)
- `DB_USER`: Database user (default: postgres)
- `DB_PASSWORD`: Database password (default: postgres)
- `MODEL_NAME`: HuggingFace model name (default: Salesforce/blip-image-captioning-base)

## Usage Example

```bash
# Caption all uncaptioned shots (limit 50)
curl -X POST http://localhost:5556/harvest \
  -H "Content-Type: application/json" \
  -d '{"limit": 50}'

# Caption specific shot
curl -X POST http://localhost:5556/caption/shot/123

# Caption uploaded image
curl -X POST http://localhost:5556/caption \
  -F "image=@photo.jpg"
```

## Performance

- First run downloads ~990MB model
- GPU recommended but works on CPU
- Processing time: ~1-2 seconds per image on CPU, ~0.3s on GPU

## Database Schema

Updates the `shots` table:
```sql
UPDATE shots
SET comment = '<generated caption>'
WHERE shot_id = <id>
```
