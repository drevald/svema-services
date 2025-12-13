# Face Recognition Service

Python microservice for face detection, encoding, and clustering using the `face_recognition` library (built on dlib).

## Features

- Face detection in images
- Face encoding generation (128-d face embeddings)
- Face clustering using DBSCAN
- Simple REST API
- No native library dependency hell!

## API Endpoints

### Health Check
```
GET /health
```

Returns service health status.

###  Detect Faces
```
POST /detect
Content-Type: multipart/form-data
```

Upload an image and get back face locations and encodings.

**Request:**
- `image`: Image file (JPEG, PNG)

**Response:**
```json
{
  "faces": [
    {
      "location": {
        "top": 100,
        "right": 200,
        "bottom": 300,
        "left": 100,
        "width": 100,
        "height": 200
      },
      "encoding": [0.123, -0.456, ...]
    }
  ],
  "count": 1
}
```

### Cluster Faces
```
POST /cluster
Content-Type: application/json
```

Cluster face encodings to group similar faces together.

**Request:**
```json
{
  "encodings": [
    [0.123, -0.456, ...],
    [0.789, -0.012, ...],
    ...
  ]
}
```

**Response:**
```json
{
  "clusters": [0, 0, 1, -1, 0],
  "unique_clusters": 2
}
```

Cluster labels:
- `-1`: Noise/outliers (faces that don't match any group)
- `0, 1, 2...`: Cluster IDs for grouped faces

## Running

### With Docker Compose
```bash
docker compose up -d --build
```

### Local Development
```bash
pip install -r requirements.txt
python app.py
```

## Integration with Svema

The main Svema application can call this service via HTTP:

```csharp
// Example C# integration
var client = new HttpClient();
var content = new MultipartFormDataContent();
content.Add(new ByteArrayContent(imageBytes), "image", "photo.jpg");

var response = await client.PostAsync("http://face-recognition:5000/detect", content);
var result = await response.Content.ReadAsStringAsync();
```

## Technology Stack

- **Python 3.11**
- **face_recognition** - Face detection and encoding (based on dlib)
- **scikit-learn** - DBSCAN clustering
- **Flask** - REST API framework
- **Pillow** - Image processing
- **NumPy** - Numerical operations

## Why Python?

Moving face detection from C#/OpenCvSharp to Python eliminates:
- Native library version conflicts
- Platform-specific compilation issues
- FFmpeg/tesseract dependency management
- Complex NuGet package compatibility

Python's ML/CV ecosystem is mature, well-maintained, and "just works".
