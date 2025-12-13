from flask import Flask, request, jsonify
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import io
import time
from datetime import datetime
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration from environment
DB_HOST = os.getenv('DB_HOST', 'svema-postgres-1')
DB_PORT = os.getenv('DB_PORT', '5433')
DB_NAME = os.getenv('DB_NAME', 'svema')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')

# Model configuration - using original BLIP base model for lower memory usage
MODEL_NAME = os.getenv('MODEL_NAME', 'Salesforce/blip-image-captioning-base')
device = "cuda" if torch.cuda.is_available() else "cpu"

# Bot user configuration - extract model name for bot username
model_short_name = MODEL_NAME.split('/')[-1]  # e.g., "blip2-flan-t5-xl"
BOT_USERNAME = f"bot-{model_short_name}"
BOT_EMAIL = f"{BOT_USERNAME}@svema.ai"

# Path mapping configuration - convert Windows paths to Linux mount paths
# Windows: Y:\FOTO\... -> Linux: /storage/user_1/FOTO/...
WINDOWS_PATH_PREFIX = os.getenv('WINDOWS_PATH_PREFIX', 'Y:\\')
LINUX_PATH_PREFIX = os.getenv('LINUX_PATH_PREFIX', '/storage/user_1/')

def convert_windows_path_to_linux(windows_path):
    """Convert Windows path (Y:\FOTO\...) to Linux mount path (/storage/user_1/FOTO/...)"""
    if not windows_path:
        return None

    # Replace Windows path prefix with Linux mount prefix
    linux_path = windows_path.replace(WINDOWS_PATH_PREFIX, LINUX_PATH_PREFIX)
    # Convert backslashes to forward slashes
    linux_path = linux_path.replace('\\', '/')

    return linux_path

# Load model at startup
logger.info(f"Loading model {MODEL_NAME} on device {device}...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
# BLIP base model is small enough to run on CPU without quantization
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
logger.info("Model loaded successfully")


def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def get_or_create_bot_user(cursor):
    """Get or create bot user in database"""
    # Check if bot user exists
    cursor.execute("""
        SELECT id FROM users WHERE username = %s
    """, (BOT_USERNAME,))

    result = cursor.fetchone()
    if result:
        return result['id'] if isinstance(result, dict) else result[0]

    # Create bot user if it doesn't exist
    cursor.execute("""
        INSERT INTO users (username, password_hash, email)
        VALUES (%s, %s, %s)
        RETURNING id
    """, (BOT_USERNAME, '', BOT_EMAIL))

    bot_id = cursor.fetchone()[0]
    logger.info(f"Created bot user: {BOT_USERNAME} with ID {bot_id}")
    return bot_id


def generate_caption(image_bytes):
    """Generate caption for image using BLIP model"""
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        logger.info(f"Loaded image: size={image.size}, mode={image.mode}")

        # Process image (padding is handled automatically for single images)
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


@app.route('/caption/shot/<int:shot_id>', methods=['POST'])
def caption_shot(shot_id):
    """
    Generate caption for a specific shot by ID and add as comment.
    Returns: JSON with caption and comment ID
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get or create bot user
        bot_user_id = get_or_create_bot_user(cursor)

        # Fetch shot data - using 'id', 'fullscreen', and 'preview' (blob data)
        cursor.execute("""
            SELECT id, fullscreen, preview
            FROM shots
            WHERE id = %s
        """, (shot_id,))

        shot = cursor.fetchone()

        if not shot:
            cursor.close()
            conn.close()
            return jsonify({"error": "Shot not found"}), 404

        # Use fullscreen (better quality) with fallback to preview
        image_bytes = bytes(shot['fullscreen']) if shot['fullscreen'] else (bytes(shot['preview']) if shot['preview'] else None)
        if not image_bytes:
            cursor.close()
            conn.close()
            return jsonify({"error": f"No image data for shot {shot_id}"}), 404

        # Generate caption
        caption = generate_caption(image_bytes)

        # Insert comment into shot_comments table
        cursor.execute("""
            INSERT INTO shot_comments (author_id, author_username, shot_id, time, text)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (bot_user_id, BOT_USERNAME, shot_id, datetime.utcnow(), caption))

        comment_id = cursor.fetchone()['id']
        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"Added caption to shot {shot_id}: {caption}")
        return jsonify({
            "shot_id": shot_id,
            "caption": caption,
            "comment_id": comment_id,
            "author": BOT_USERNAME
        }), 200

    except Exception as e:
        logger.error(f"Error captioning shot {shot_id}: {e}")
        if conn:
            conn.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/harvest', methods=['POST'])
def harvest_shots():
    """
    Process all shots without AI comments and generate captions.
    Optional parameters:
    - limit: maximum number of shots to process (default: 100)
    - album_id: only process shots from specific album
    Returns: JSON with processing statistics
    """
    limit = request.json.get('limit', 100) if request.json else 100
    album_id = request.json.get('album_id') if request.json else None

    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get or create bot user
        bot_user_id = get_or_create_bot_user(cursor)

        # Build query - find shots without AI-generated comments, get fullscreen and preview data
        query = """
            SELECT s.id, s.fullscreen, s.preview
            FROM shots s
            WHERE NOT EXISTS (
                SELECT 1 FROM shot_comments sc
                WHERE sc.shot_id = s.id AND sc.author_id = %s
            )
            AND (s.fullscreen IS NOT NULL OR s.preview IS NOT NULL)
        """
        params = [bot_user_id]

        if album_id:
            query += " AND s.album_id = %s"
            params.append(album_id)

        query += " ORDER BY s.id DESC LIMIT %s"
        params.append(limit)

        cursor.execute(query, params)
        shots = cursor.fetchall()

        processed = 0
        failed = 0
        skipped = 0
        results = []

        for shot in shots:
            shot_id = shot['id']

            try:
                # Use fullscreen (better quality) with fallback to preview
                image_bytes = bytes(shot['fullscreen']) if shot['fullscreen'] else (bytes(shot['preview']) if shot['preview'] else None)
                if not image_bytes:
                    logger.warning(f"No image data for shot {shot_id}")
                    failed += 1
                    continue

                caption = generate_caption(image_bytes)

                # Insert comment
                cursor.execute("""
                    INSERT INTO shot_comments (author_id, author_username, shot_id, time, text)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (bot_user_id, BOT_USERNAME, shot_id, datetime.utcnow(), caption))

                comment_id = cursor.fetchone()['id']
                conn.commit()
                processed += 1
                results.append({
                    "shot_id": shot_id,
                    "comment_id": comment_id,
                    "caption": caption
                })

                logger.info(f"Processed shot {shot_id}: {caption}")

            except Exception as e:
                logger.error(f"Error processing shot {shot_id}: {e}")
                conn.rollback()
                failed += 1
                continue

        cursor.close()
        conn.close()

        return jsonify({
            "total_found": len(shots),
            "processed": processed,
            "failed": failed,
            "bot_user": BOT_USERNAME,
            "results": results[:10]  # Return first 10 for brevity
        }), 200

    except Exception as e:
        logger.error(f"Error harvesting shots: {e}")
        if conn:
            conn.rollback()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5556)
