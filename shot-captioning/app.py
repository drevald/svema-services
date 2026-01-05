from flask import Flask, request, jsonify, render_template, send_from_directory
import io
from PIL import Image
import logging
import os
import base64
import requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://192.168.0.134:11434')
MODEL_NAME = os.getenv('MODEL_NAME', 'qwen3-vl:8b')

# Test connection to Ollama
logger.info(f"Connecting to Ollama at {OLLAMA_BASE_URL}...")
try:
    response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    if response.status_code == 200:
        available_models = [model['name'] for model in response.json().get('models', [])]
        logger.info(f"Successfully connected to Ollama. Available models: {available_models}")
        if MODEL_NAME not in available_models and f"{MODEL_NAME}:latest" not in available_models:
            logger.warning(f"Model {MODEL_NAME} not found in available models. You may need to pull it first.")
    else:
        logger.warning(f"Could not retrieve model list from Ollama: {response.status_code}")
except Exception as e:
    logger.warning(f"Could not connect to Ollama at startup: {e}")
    logger.info("Service will attempt to connect when processing requests.")

logger.info(f"Using model: {MODEL_NAME}")


def generate_caption(image_bytes):
    """Generate caption for image using Ollama vision model"""
    try:
        # Convert image bytes to base64 for Ollama API
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Load image for logging
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        logger.info(f"Loaded image: size={image.size}, mode={image.mode}")

        # Call Ollama API with vision model
        payload = {
            "model": MODEL_NAME,
            "prompt": "Describe this image in detail. Focus on the main subjects, actions, setting, and mood.",
            "images": [image_base64],
            "stream": False
        }

        logger.info(f"Sending request to Ollama at {OLLAMA_BASE_URL}/api/generate")
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            raise Exception(f"Ollama API returned status {response.status_code}")

        result = response.json()
        caption = result.get('response', '').strip()

        logger.info(f"Generated caption: {caption}")
        return caption
    except Exception as e:
        logger.error(f"Error generating caption: {e}", exc_info=True)
        raise


@app.route('/', methods=['GET', 'POST'])
def index():
    """Web interface for image captioning"""
    if request.method == 'GET':
        return render_template('index.html', model_name=MODEL_NAME, ollama_url=OLLAMA_BASE_URL)

    # POST request - handle image upload
    if 'image' not in request.files:
        return render_template('index.html', model_name=MODEL_NAME, ollama_url=OLLAMA_BASE_URL,
                             error="No image file provided")

    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('index.html', model_name=MODEL_NAME, ollama_url=OLLAMA_BASE_URL,
                             error="No file selected")

    try:
        image_bytes = image_file.read()
        caption = generate_caption(image_bytes)

        # Convert image to base64 for display
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        return render_template('index.html', model_name=MODEL_NAME, ollama_url=OLLAMA_BASE_URL,
                             caption=caption, image_data=image_base64)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return render_template('index.html', model_name=MODEL_NAME, ollama_url=OLLAMA_BASE_URL,
                             error=f"Error processing image: {str(e)}")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": MODEL_NAME, "ollama_url": OLLAMA_BASE_URL}), 200


@app.route('/caption', methods=['POST'])
def caption_image():
    """
    Generate caption for a single image.
    Expected input: multipart/form-data with 'image' file
    Returns: JSON with caption
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    try:
        caption = generate_caption(image_bytes)
        return jsonify({"caption": caption}), 200
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5556)
