#!/usr/bin/env python3
import requests
from PIL import Image, ImageDraw
import io

# Create a simple test image (100x100 red square as placeholder)
img = Image.new('RGB', (100, 100), color='white')
draw = ImageDraw.Draw(img)
draw.rectangle([30, 30, 70, 70], fill='red')

# Save to bytes
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

# Test /detect endpoint
print("Testing /detect endpoint...")
response = requests.post(
    'http://localhost:5555/detect',
    files={'image': ('test.jpg', img_bytes, 'image/jpeg')}
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Test /cluster endpoint with sample encodings
print("\nTesting /cluster endpoint...")
sample_encodings = [
    [0.1] * 128,  # 128-d encoding
    [0.1] * 128,  # Similar encoding (should cluster together)
    [0.9] * 128,  # Different encoding
]

response = requests.post(
    'http://localhost:5555/cluster',
    json={'encodings': sample_encodings}
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
