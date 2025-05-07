# scripts/feature_extraction.py
import os
import numpy as np
import cv2
import mediapipe as mp
import pickle
import psycopg2
import io
from psycopg2 import Binary
import pandas as pd

# Database connection parameters
DB_PARAMS = {
    "host": "localhost",
    "database": "child_face_db",
    "user": "postgres",
    "password": "tranphuong"  # Thay bằng mật khẩu của bạn
}

# Khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection


def extract_mediapipe_face_landmarks(image_path):
    """Extract facial landmarks using MediaPipe Face Mesh."""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return None

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Face Mesh
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,  # Assuming one face per image
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:

            results = face_mesh.process(image_rgb)

            if not results.multi_face_landmarks:
                print(f"No face detected in {image_path}")
                return None

            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]

            # Convert landmarks to a numpy array
            landmarks_array = []
            for landmark in face_landmarks.landmark:
                landmarks_array.extend([landmark.x, landmark.y, landmark.z])

            return np.array(landmarks_array, dtype=np.float32)

    except Exception as e:
        print(f"Error extracting landmarks from {image_path}: {e}")
        return None


def extract_mediapipe_face_detection(image_path):
    """Extract face detection features using MediaPipe Face Detection."""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return None

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Face Detection
        with mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for closer faces, 1 for farther faces
                min_detection_confidence=0.5) as face_detection:

            results = face_detection.process(image_rgb)

            if not results.detections:
                print(f"No face detected in {image_path}")
                return None

            # Get first face detection
            detection = results.detections[0]

            # Extract face bounding box and keypoints
            bbox = detection.location_data.relative_bounding_box
            keypoints = []

            # Add normalized bounding box coordinates
            bbox_array = [bbox.xmin, bbox.ymin, bbox.width, bbox.height]

            # Add six face keypoints (right eye, left eye, nose tip, mouth center, right ear tragion, left ear tragion)
            for i in range(6):
                kp = detection.location_data.relative_keypoints[i]
                keypoints.extend([kp.x, kp.y])

            # Combine bbox and keypoints into a single feature vector
            features = np.array(bbox_array + keypoints, dtype=np.float32)
            return features

    except Exception as e:
        print(f"Error extracting face detection from {image_path}: {e}")
        return None


def extract_facial_attributes(image_path):
    """
    Extract facial attributes like skin color, face shape, etc.
    This is a simplified version and can be expanded for more sophisticated analysis.
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return None

        # Convert to RGB for better color analysis
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get face area using MediaPipe Face Detection
        with mp_face_detection.FaceDetection(
                min_detection_confidence=0.5) as face_detection:

            results = face_detection.process(image_rgb)

            if not results.detections:
                print(f"No face detected in {image_path}")
                return None

            # Get face bounding box
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            h, w, _ = image.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Ensure valid coordinates
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)

            # Extract face region
            face_region = image_rgb[y:y + height, x:x + width]

            if face_region.size == 0:
                print(f"Invalid face region in {image_path}")
                return None

            # Compute average skin color (simplified)
            average_color = np.mean(face_region, axis=(0, 1))

            # Compute face shape features (simplified - aspect ratio of face)
            face_aspect_ratio = height / width if width > 0 else 0

            # Combine into feature vector
            features = np.array([
                average_color[0], average_color[1], average_color[2],
                face_aspect_ratio,
                width * height  # Face area in pixels
            ], dtype=np.float32)

            return features

    except Exception as e:
        print(f"Error extracting facial attributes from {image_path}: {e}")
        return None


def extract_hog_features(image_path, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    """Extract HOG (Histogram of Oriented Gradients) features."""
    try:
        # Load image and convert to grayscale
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate HOG features
        win_size = gray.shape
        block_stride = (cell_size[0] // 2, cell_size[1] // 2)

        hog = cv2.HOGDescriptor(
            win_size,
            block_size,
            block_stride,
            cell_size,
            nbins
        )

        hog_features = hog.compute(gray)
        return hog_features.flatten()

    except Exception as e:
        print(f"Error extracting HOG features from {image_path}: {e}")
        return None


def store_features_in_db(image_id, feature_type, feature_vector):
    """Store feature vector in the database."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()

        # Convert numpy array to binary data
        vector_bytes = io.BytesIO()
        np.save(vector_bytes, feature_vector)
        vector_bytes.seek(0)

        # Check if features already exist for this image and feature type
        cursor.execute(
            """
            SELECT id
            FROM features
            WHERE image_id = %s AND feature_type = %s
            """,
            (image_id, feature_type)
        )

        exists = cursor.fetchone()

        if exists:
            # Update existing features
            cursor.execute(
                """
                UPDATE features
                SET feature_vector = %s
                WHERE image_id = %s AND feature_type = %s
                """,
                (Binary(vector_bytes.read()), image_id, feature_type)
            )
        else:
            # Insert new features
            cursor.execute(
                """
                INSERT INTO features (image_id, feature_type, feature_vector)
                VALUES (%s, %s, %s)
                """,
                (image_id, feature_type, Binary(vector_bytes.read()))
            )

        conn.commit()
        cursor.close()
        conn.close()

        return True

    except Exception as e:
        print(f"Error storing features in database: {e}")
        return False


def extract_and_store_features(processed_dir, metadata_path):
    """Extract features for all processed images and store in database."""
    # Load metadata
    metadata = pd.read_csv(metadata_path)

    success_count = 0

    # Initialize MediaPipe solutions (outside loop for better performance)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        with mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5) as face_detection:

            for _, row in metadata.iterrows():
                image_id = row['id']
                image_path = row['path']

                print(f"Processing image: {image_path}")

                # Extract MediaPipe face landmarks
                landmarks = extract_mediapipe_face_landmarks(image_path)
                if landmarks is not None:
                    if store_features_in_db(image_id, 'mediapipe_landmarks', landmarks):
                        success_count += 1

                # Extract MediaPipe face detection features
                face_detection_features = extract_mediapipe_face_detection(image_path)
                if face_detection_features is not None:
                    store_features_in_db(image_id, 'mediapipe_detection', face_detection_features)

                # Extract facial attributes
                facial_attributes = extract_facial_attributes(image_path)
                if facial_attributes is not None:
                    store_features_in_db(image_id, 'facial_attributes', facial_attributes)

                # Extract HOG features
                hog_features = extract_hog_features(image_path)
                if hog_features is not None:
                    store_features_in_db(image_id, 'hog', hog_features)

    print(f"Successfully extracted and stored features for {success_count} images")


if __name__ == "__main__":
    processed_dir = "../data/processed_images"  # Đường dẫn tới thư mục chứa ảnh đã xử lý
    metadata_path = os.path.join(processed_dir, 'metadata.csv')
    extract_and_store_features(processed_dir, metadata_path)