import pytest
import io
import json
from PIL import Image, ImageDraw
from app import app


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def create_face_image():
    """
    Create a simple synthetic face-like image for testing.
    This creates a basic representation with facial features.
    """
    # Create blank image
    img = Image.new('RGB', (400, 500), color='peachpuff')
    draw = ImageDraw.Draw(img)

    # Draw face outline (ellipse)
    draw.ellipse([100, 80, 300, 380], fill='peachpuff', outline='black', width=3)

    # Draw left eye
    draw.ellipse([140, 180, 180, 220], fill='white', outline='black', width=2)
    draw.ellipse([150, 190, 170, 210], fill='black')

    # Draw right eye
    draw.ellipse([220, 180, 260, 220], fill='white', outline='black', width=2)
    draw.ellipse([230, 190, 250, 210], fill='black')

    # Draw nose
    draw.polygon([(200, 230), (185, 280), (215, 280)], fill='tan', outline='black')

    # Draw mouth
    draw.arc([160, 290, 240, 340], 0, 180, fill='black', width=3)

    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    return img_bytes


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'


def test_detect_face(client):
    """Test face detection with synthetic face image"""
    img_bytes = create_face_image()

    response = client.post('/detect', data={'image': (img_bytes, 'test.jpg')})
    assert response.status_code == 200

    data = json.loads(response.data)

    # Should detect at least one face
    assert data['count'] >= 1, "Should detect at least one face"
    assert len(data['faces']) >= 1, "Should return at least one face"

    # Check first face has required fields
    face = data['faces'][0]
    assert 'location' in face
    assert 'encoding' in face

    location = face['location']
    assert 'top' in location
    assert 'right' in location
    assert 'bottom' in location
    assert 'left' in location
    assert 'width' in location
    assert 'height' in location

    # Face should be roughly in the center of the image (400x500)
    # Allow generous tolerance since our synthetic face may not be perfect
    assert 50 < location['left'] < 250, f"Left coordinate {location['left']} should be roughly in center"
    assert 150 < location['right'] < 350, f"Right coordinate {location['right']} should be roughly in center"
    assert 50 < location['top'] < 300, f"Top coordinate {location['top']} should be in upper portion"
    assert 200 < location['bottom'] < 450, f"Bottom coordinate {location['bottom']} should be in lower portion"

    # Face width and height should be reasonable
    assert 50 < location['width'] < 300, f"Face width {location['width']} should be reasonable"
    assert 100 < location['height'] < 400, f"Face height {location['height']} should be reasonable"

    # Encoding should be a list of 128 floats
    assert isinstance(face['encoding'], list)
    assert len(face['encoding']) == 128, "Face encoding should have 128 dimensions"


def test_detect_no_image(client):
    """Test detection without providing image"""
    response = client.post('/detect', data={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_cluster_endpoint(client):
    """Test clustering endpoint with sample encodings"""
    # Create sample encodings (128-dimensional vectors)
    sample_encodings = [
        [0.1] * 128,  # Similar faces
        [0.11] * 128,
        [0.9] * 128,  # Different face
    ]

    response = client.post('/cluster',
                          data=json.dumps({'encodings': sample_encodings}),
                          content_type='application/json')

    assert response.status_code == 200
    data = json.loads(response.data)

    assert 'clusters' in data
    assert 'unique_clusters' in data
    assert len(data['clusters']) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
