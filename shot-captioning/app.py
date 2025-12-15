from flask import Flask, request, jsonify
import io
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import logging
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'Salesforce/blip2-opt-2.7b')
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model at startup
logger.info(f"Loading model {MODEL_NAME} on device {device}...")
processor = Blip2Processor.from_pretrained(MODEL_NAME)
model = Blip2ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
logger.info("Model loaded successfully")


def generate_caption(image_bytes):
    """Generate caption for image using BLIP model"""
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        logger.info(f"Loaded image: size={image.size}, mode={image.mode}")

        # Process image
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Generate caption
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50, num_beams=5)

        caption = processor.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated caption: {caption}")
        return caption
    except Exception as e:
        logger.error(f"Error generating caption: {e}", exc_info=True)
        raise


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": MODEL_NAME, "device": device}), 200


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
