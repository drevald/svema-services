import pytest
import io
import json
from PIL import Image, ImageDraw, ImageFont
from app import app


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def create_simple_image(subject, color):
    """
    Create a simple synthetic image with basic shapes representing objects.

    Args:
        subject: Type of object ('house', 'tree', 'dog', 'cat')
        color: Background color
    """
    img = Image.new('RGB', (400, 400), color=color)
    draw = ImageDraw.Draw(img)

    if subject == 'house':
        # Draw a house
        # Base
        draw.rectangle([100, 200, 300, 350], fill='brown', outline='black', width=3)
        # Roof
        draw.polygon([(100, 200), (200, 100), (300, 200)], fill='red', outline='black')
        # Door
        draw.rectangle([170, 270, 230, 350], fill=(101, 67, 33), outline='black', width=2)
        # Window
        draw.rectangle([120, 230, 170, 280], fill=(173, 216, 230), outline='black', width=2)
        draw.rectangle([230, 230, 280, 280], fill=(173, 216, 230), outline='black', width=2)

    elif subject == 'tree':
        # Draw a tree
        # Trunk
        draw.rectangle([180, 250, 220, 370], fill='brown', outline='black', width=2)
        # Leaves (multiple circles for foliage)
        draw.ellipse([120, 150, 280, 280], fill='green', outline='darkgreen', width=2)
        draw.ellipse([140, 120, 260, 240], fill='green', outline='darkgreen', width=2)
        draw.ellipse([160, 100, 240, 200], fill='green', outline='darkgreen', width=2)

    elif subject == 'dog':
        # Draw a simple dog
        # Body
        draw.ellipse([120, 200, 280, 300], fill='brown', outline='black', width=2)
        # Head
        draw.ellipse([240, 150, 340, 250], fill='brown', outline='black', width=2)
        # Ears
        draw.ellipse([240, 140, 270, 200], fill='brown', outline='black', width=2)
        draw.ellipse([310, 140, 340, 200], fill='brown', outline='black', width=2)
        # Legs
        draw.rectangle([140, 280, 160, 350], fill='brown', outline='black', width=2)
        draw.rectangle([180, 280, 200, 350], fill='brown', outline='black', width=2)
        draw.rectangle([220, 280, 240, 350], fill='brown', outline='black', width=2)
        draw.rectangle([260, 280, 280, 350], fill='brown', outline='black', width=2)
        # Tail
        draw.arc([100, 190, 150, 240], 180, 90, fill='black', width=4)

    elif subject == 'cat':
        # Draw a simple cat
        # Body
        draw.ellipse([140, 220, 260, 300], fill='orange', outline='black', width=2)
        # Head
        draw.ellipse([220, 160, 320, 260], fill='orange', outline='black', width=2)
        # Ears (triangular)
        draw.polygon([(230, 160), (240, 130), (250, 160)], fill='orange', outline='black')
        draw.polygon([(290, 160), (300, 130), (310, 160)], fill='orange', outline='black')
        # Legs
        draw.rectangle([160, 280, 180, 340], fill='orange', outline='black', width=2)
        draw.rectangle([200, 280, 220, 340], fill='orange', outline='black', width=2)
        # Tail
        draw.arc([80, 240, 160, 320], 270, 90, fill='black', width=4)

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
    assert 'model' in data
    assert 'device' in data


def test_caption_no_image(client):
    """Test captioning without providing image"""
    response = client.post('/caption', data={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


@pytest.mark.slow
def test_caption_house(client):
    """Test captioning a simple house image"""
    img_bytes = create_simple_image('house', 'skyblue')

    response = client.post('/caption', data={'image': (img_bytes, 'house.jpg')})
    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'caption' in data
    assert isinstance(data['caption'], str)
    assert len(data['caption']) > 0

    # Caption should be somewhat reasonable - at least return something
    print(f"House caption: {data['caption']}")


@pytest.mark.slow
def test_caption_tree(client):
    """Test captioning a simple tree image"""
    img_bytes = create_simple_image('tree', 'lightblue')

    response = client.post('/caption', data={'image': (img_bytes, 'tree.jpg')})
    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'caption' in data
    assert isinstance(data['caption'], str)
    assert len(data['caption']) > 0

    print(f"Tree caption: {data['caption']}")


@pytest.mark.slow
def test_caption_dog(client):
    """Test captioning a simple dog image"""
    img_bytes = create_simple_image('dog', 'white')

    response = client.post('/caption', data={'image': (img_bytes, 'dog.jpg')})
    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'caption' in data
    assert isinstance(data['caption'], str)
    assert len(data['caption']) > 0

    print(f"Dog caption: {data['caption']}")


@pytest.mark.slow
def test_caption_cat(client):
    """Test captioning a simple cat image"""
    img_bytes = create_simple_image('cat', 'white')

    response = client.post('/caption', data={'image': (img_bytes, 'cat.jpg')})
    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'caption' in data
    assert isinstance(data['caption'], str)
    assert len(data['caption']) > 0

    print(f"Cat caption: {data['caption']}")


if __name__ == '__main__':
    # Run tests with verbose output
    # Skip slow tests by default (model loading takes time)
    # To run all tests: pytest test_app.py -v -m ""
    pytest.main([__file__, '-v', '-m', 'not slow'])
