from flask import Flask, request, jsonify
import face_recognition
import numpy as np
from PIL import Image
import io
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use CNN model for better accuracy (slower but fewer false positives)
# Set USE_CNN_MODEL=true in environment to enable
USE_CNN_MODEL = os.getenv('USE_CNN_MODEL', 'false').lower() == 'true'

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/detect', methods=['POST'])
def detect_faces():
    """
    Detect faces in an image and return face locations and encodings.

    Expected input: multipart/form-data with 'image' file
    Returns: JSON with face locations and encodings
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()

        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image.convert('RGB'))
        img_height, img_width = image_array.shape[:2]

        # Detect faces - CNN model is more accurate but slower
        model = 'cnn' if USE_CNN_MODEL else 'hog'
        face_locations = face_recognition.face_locations(image_array, model=model, number_of_times_to_upsample=1)
        logger.info(f"Using {model.upper()} model for face detection")

        if not face_locations:
            return jsonify({
                "faces": [],
                "count": 0
            }), 200

        # Get face landmarks for validation (detects facial features)
        face_landmarks_list = face_recognition.face_landmarks(image_array, face_locations)

        # Filter out faces that are likely false positives
        filtered_locations = []
        for idx, (top, right, bottom, left) in enumerate(face_locations):
            width = right - left
            height = bottom - top

            # Filter by size - minimum 1% of image, maximum 95%
            min_size = min(img_width, img_height) * 0.01  # Increased from 0.005 to reduce false positives
            max_size = min(img_width, img_height) * 0.95

            if width < min_size or height < min_size:
                logger.debug(f"Filtered out face: too small ({width}x{height})")
                continue
            if width > max_size or height > max_size:
                logger.debug(f"Filtered out face: too large ({width}x{height})")
                continue

            # Filter by aspect ratio - real faces are 0.7 to 1.3 (stricter than before)
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                logger.debug(f"Filtered out face: bad aspect ratio ({aspect_ratio:.2f})")
                continue

            # Validate facial landmarks - must have key facial features
            if idx < len(face_landmarks_list):
                landmarks = face_landmarks_list[idx]
                # Check for essential facial features
                required_features = ['left_eye', 'right_eye', 'nose_tip', 'top_lip', 'bottom_lip']
                has_all_features = all(feature in landmarks and len(landmarks[feature]) > 0 for feature in required_features)

                if not has_all_features:
                    logger.debug(f"Filtered out face: missing facial features")
                    continue

                # Check if eyes are properly positioned (not too close, not too far)
                left_eye = np.array(landmarks['left_eye'])
                right_eye = np.array(landmarks['right_eye'])
                eye_distance = np.linalg.norm(left_eye.mean(axis=0) - right_eye.mean(axis=0))

                # Eye distance should be roughly 30-70% of face width
                eye_distance_ratio = eye_distance / width if width > 0 else 0
                if eye_distance_ratio < 0.25 or eye_distance_ratio > 0.75:
                    logger.debug(f"Filtered out face: abnormal eye distance ratio ({eye_distance_ratio:.2f})")
                    continue

            filtered_locations.append((top, right, bottom, left))

        if not filtered_locations:
            logger.info(f"Detected {len(face_locations)} face(s) but all filtered out")
            return jsonify({
                "faces": [],
                "count": 0
            }), 200

        # Get face encodings for filtered faces
        face_encodings = face_recognition.face_encodings(image_array, filtered_locations)

        # Format response
        faces = []
        for location, encoding in zip(filtered_locations, face_encodings):
            top, right, bottom, left = location
            faces.append({
                "location": {
                    "top": int(top),
                    "right": int(right),
                    "bottom": int(bottom),
                    "left": int(left),
                    "width": int(right - left),
                    "height": int(bottom - top)
                },
                "encoding": encoding.tolist()
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

        # Use face_recognition's built-in clustering
        from sklearn.cluster import DBSCAN

        # DBSCAN clustering with tolerance for face distance
        clustering = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
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
