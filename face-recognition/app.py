from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import logging
import os
import base64
import cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize InsightFace
try:
    import insightface
    from insightface.app import FaceAnalysis

    # Initialize face analysis model
    face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("InsightFace initialized successfully with buffalo_l model")
except Exception as e:
    logger.error(f"Failed to initialize InsightFace: {e}")
    face_app = None


def detect_faces_with_insightface(image_bytes):
    """
    Detect faces using InsightFace.
    Returns list of face detections with locations and embeddings.
    """
    try:
        if face_app is None:
            logger.error("InsightFace not initialized")
            return []

        # Convert image bytes to numpy array
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(image)

        # InsightFace expects BGR format
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Detect faces
        faces = face_app.get(img_bgr)

        logger.info(f"InsightFace detected {len(faces)} face(s)")

        # Convert to our format
        face_detections = []
        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(int)

            # bbox format: [x1, y1, x2, y2]
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]

            # Ensure coordinates are within image bounds
            left = max(0, min(left, image.width))
            top = max(0, min(top, image.height))
            right = max(0, min(right, image.width))
            bottom = max(0, min(bottom, image.height))

            # Get face embedding (512-dimensional vector)
            embedding = face.embedding

            face_detections.append({
                'location': (top, right, bottom, left),
                'embedding': embedding,
                'confidence': float(face.det_score)
            })

            logger.info(f"Face {idx+1}: bbox=({left},{top},{right},{bottom}), confidence={face.det_score:.3f}")

        return face_detections

    except Exception as e:
        logger.error(f"Error detecting faces with InsightFace: {e}", exc_info=True)
        return []


def draw_faces_on_image(image_bytes, face_locations, confidences=None):
    """Draw bounding boxes around detected faces"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    draw = ImageDraw.Draw(image)

    for idx, (top, right, bottom, left) in enumerate(face_locations, 1):
        # Draw rectangle
        draw.rectangle([(left, top), (right, bottom)], outline='#00ff00', width=3)

        # Draw label with face number and confidence
        if confidences and idx <= len(confidences):
            label = f"Face {idx} ({confidences[idx-1]:.2f})"
        else:
            label = f"Face {idx}"

        # Use default font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Draw label background
        bbox = draw.textbbox((left, top - 25), label, font=font)
        draw.rectangle(bbox, fill='#00ff00')
        draw.text((left, top - 25), label, fill='black', font=font)

    # Convert back to bytes
    output = io.BytesIO()
    image.save(output, format='JPEG')
    return output.getvalue()


@app.route('/', methods=['GET', 'POST'])
def index():
    """Web interface for face detection"""
    if request.method == 'GET':
        return render_template('index.html', model_name='InsightFace (buffalo_l)')

    # POST request - handle image upload
    if 'image' not in request.files:
        return render_template('index.html', model_name='InsightFace (buffalo_l)',
                             error="No image file provided")

    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('index.html', model_name='InsightFace (buffalo_l)',
                             error="No file selected")

    try:
        image_bytes = image_file.read()

        # Detect faces using InsightFace
        faces = detect_faces_with_insightface(image_bytes)
        face_locations = [face['location'] for face in faces]
        confidences = [face['confidence'] for face in faces]

        # Draw boxes on image
        if face_locations:
            result_image_bytes = draw_faces_on_image(image_bytes, face_locations, confidences)
        else:
            result_image_bytes = image_bytes

        # Convert to base64
        image_base64 = base64.b64encode(result_image_bytes).decode('utf-8')

        return render_template('index.html', model_name='InsightFace (buffalo_l)',
                             face_count=len(face_locations),
                             image_data=image_base64,
                             faces=[{
                                 'confidence': f"{c:.3f}",
                                 'location': f"({l[3]}, {l[0]}, {l[1]}, {l[2]})"
                             } for c, l in zip(confidences, face_locations)])

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        return render_template('index.html', model_name='InsightFace (buffalo_l)',
                             error=f"Error processing image: {str(e)}")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": "InsightFace buffalo_l"}), 200


@app.route('/detect', methods=['POST'])
def detect_faces():
    """
    Detect faces in an image and return face locations and encodings.

    Expected input: multipart/form-data with 'image' file
    Returns: JSON with face locations and embeddings
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()

        # Detect faces using InsightFace
        faces_data = detect_faces_with_insightface(image_bytes)

        if not faces_data:
            return jsonify({
                "faces": [],
                "count": 0
            }), 200

        # Convert to API response format
        faces = []
        for face_info in faces_data:
            top, right, bottom, left = face_info['location']

            faces.append({
                "location": {
                    "top": int(top),
                    "right": int(right),
                    "bottom": int(bottom),
                    "left": int(left),
                    "width": int(right - left),
                    "height": int(bottom - top)
                },
                "encoding": face_info['embedding'].tolist(),
                "confidence": face_info['confidence']
            })

        logger.info(f"Detected {len(faces)} face(s)")

        return jsonify({
            "faces": faces,
            "count": len(faces)
        }), 200

    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/cluster', methods=['POST'])
def cluster_faces():
    """
    Cluster face encodings using DBSCAN.

    Expected input: JSON with array of encodings
    Returns: JSON with cluster assignments
    """
    try:
        data = request.get_json()

        if 'encodings' not in data:
            return jsonify({"error": "No encodings provided"}), 400

        encodings = np.array(data['encodings'])

        if len(encodings) == 0:
            return jsonify({"clusters": []}), 200

        # Use DBSCAN clustering
        from sklearn.cluster import DBSCAN

        # DBSCAN clustering with tolerance for face distance
        # InsightFace embeddings work well with cosine similarity
        # eps=0.7 provides very lenient grouping to avoid splitting same person
        clustering = DBSCAN(eps=0.7, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(encodings)

        # Convert numpy int64 to Python int for JSON serialization
        clusters = [int(label) for label in labels]

        logger.info(f"Clustered {len(encodings)} faces into {len(set(clusters))} groups")

        return jsonify({
            "clusters": clusters,
            "unique_clusters": len(set(clusters))
        }), 200

    except Exception as e:
        logger.error(f"Error clustering faces: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=False)
