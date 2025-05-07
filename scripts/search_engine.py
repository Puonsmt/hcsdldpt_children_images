# scripts/search_engine.py
import os
import numpy as np
import cv2
import mediapipe as mp
import pickle
import psycopg2
import io
from psycopg2 import Binary
import pandas as pd

# Từ feature_extraction.py
from feature_extraction import (
    extract_mediapipe_face_landmarks,
    extract_mediapipe_face_detection,
    extract_facial_attributes,
    extract_hog_features
)

# Database connection parameters
DB_PARAMS = {
    "host": "localhost",
    "database": "child_face_db",
    "user": "postgres",
    "password": "tranphuong"  # Thay bằng mật khẩu của bạn
}


def load_feature_vectors(feature_type='mediapipe_landmarks'):
    """Load all feature vectors from database."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()

        # Query to get image IDs and feature vectors
        cursor.execute(
            """
            SELECT i.id, i.file_name, i.path, f.feature_vector
            FROM images i
                     JOIN features f ON i.id = f.image_id
            WHERE f.feature_type = %s
            """,
            (feature_type,)
        )

        results = cursor.fetchall()

        feature_vectors = []
        image_ids = []
        file_names = []
        image_paths = []

        for image_id, file_name, path, feature_vector_bytes in results:
            # Convert binary data back to numpy array
            vector_bytes = io.BytesIO(feature_vector_bytes)
            feature_vector = np.load(vector_bytes, allow_pickle=True)

            feature_vectors.append(feature_vector)
            image_ids.append(image_id)
            file_names.append(file_name)
            image_paths.append(path)

        cursor.close()
        conn.close()

        return {
            'feature_vectors': np.array(feature_vectors),
            'image_ids': np.array(image_ids),
            'file_names': np.array(file_names),
            'image_paths': np.array(image_paths)
        }

    except Exception as e:
        print(f"Error loading feature vectors: {e}")
        return None


def compute_similarity(query_vector, all_vectors, metric='cosine'):
    """Compute similarity between query vector and all stored vectors."""
    if metric == 'cosine':
        # Normalize vectors for cosine similarity
        query_norm = np.linalg.norm(query_vector)
        all_norms = np.linalg.norm(all_vectors, axis=1)

        # Avoid division by zero
        query_norm = max(query_norm, 1e-10)
        all_norms = np.maximum(all_norms, 1e-10)

        # Compute dot product and then cosine similarity
        dot_products = np.dot(all_vectors, query_vector)
        similarities = dot_products / (all_norms * query_norm)

    elif metric == 'euclidean':
        # Compute Euclidean distance
        distances = np.linalg.norm(all_vectors - query_vector, axis=1)
        # Convert to similarity (smaller distance = higher similarity)
        similarities = 1 / (1 + distances)

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return similarities


def extract_query_features(query_image_path, feature_type='mediapipe_landmarks'):
    """Extract features from query image based on feature type."""
    if feature_type == 'mediapipe_landmarks':
        return extract_mediapipe_face_landmarks(query_image_path)
    elif feature_type == 'mediapipe_detection':
        return extract_mediapipe_face_detection(query_image_path)
    elif feature_type == 'facial_attributes':
        return extract_facial_attributes(query_image_path)
    elif feature_type == 'hog':
        return extract_hog_features(query_image_path)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")


def find_similar_images(query_image_path, top_k=3, feature_type='mediapipe_landmarks', metric='cosine'):
    """Find the top_k most similar images to the query image."""
    try:
        # Extract features from query image
        query_vector = extract_query_features(query_image_path, feature_type)

        if query_vector is None:
            print(f"Could not extract {feature_type} features from query image: {query_image_path}")
            return None

        # Load all feature vectors
        data = load_feature_vectors(feature_type)

        if data is None or len(data['feature_vectors']) == 0:
            print(f"Could not load {feature_type} feature vectors from database")
            return None

        # Compute similarities
        similarities = compute_similarity(query_vector, data['feature_vectors'], metric)

        # Get indices of top_k most similar images
        if metric == 'cosine':
            # For cosine similarity, higher values are better
            top_indices = np.argsort(similarities)[::-1][:top_k]
        else:
            # For distance-based metrics, lower values are better
            top_indices = np.argsort(similarities)[:top_k]

        # Return top matches
        return [
            {
                'image_id': data['image_ids'][i],
                'file_name': data['file_names'][i],
                'path': data['image_paths'][i],
                'similarity': float(similarities[i])
            }
            for i in top_indices
        ]

    except Exception as e:
        print(f"Error finding similar images: {e}")
        return None


def ensemble_search(query_image_path, top_k=3, weights=None):
    """
    Combine results from multiple feature types for better search quality.

    Parameters:
    - weights: Dictionary with feature types as keys and their weights as values
              If None, all feature types are weighted equally
    """
    # Default weights if not provided
    if weights is None:
        weights = {
            'mediapipe_landmarks': 0.4,
            'mediapipe_detection': 0.2,
            'facial_attributes': 0.2,
            'hog': 0.2
        }

    # Get results from different feature types
    results_by_type = {}
    all_image_ids = set()

    for feature_type, weight in weights.items():
        results = find_similar_images(
            query_image_path,
            top_k=top_k * 2,  # Get more results and then combine
            feature_type=feature_type
        )

        if results:
            results_by_type[feature_type] = results
            all_image_ids.update(r['image_id'] for r in results)

    if not results_by_type:
        print("No results found for any feature type")
        return None

        # Combine results
    all_results = {}

    for image_id in all_image_ids:
        all_results[image_id] = {
            'image_id': image_id,
            'weighted_scores': []
        }

    # Process each feature type's results
    for feature_type, results in results_by_type.items():
        # Get weight for this feature
        weight = weights.get(feature_type, 1.0 / len(results_by_type))

        # Find max similarity for normalization
        max_sim = max(r['similarity'] for r in results)
        max_sim = max(max_sim, 1e-10)  # Avoid division by zero

        # Add weighted normalized scores
        for r in results:
            image_id = r['image_id']
            if image_id in all_results:
                normalized_score = r['similarity'] / max_sim
                all_results[image_id]['weighted_scores'].append(normalized_score * weight)

                # Copy metadata from the first encounter
                if 'file_name' not in all_results[image_id]:
                    all_results[image_id]['file_name'] = r['file_name']
                    all_results[image_id]['path'] = r['path']

    # Calculate final scores
    for image_id, data in all_results.items():
        if data['weighted_scores']:
            data['final_score'] = sum(data['weighted_scores'])
        else:
            data['final_score'] = 0

    # Sort by final score
    sorted_results = sorted(
        [r for r in all_results.values() if 'file_name' in r],
        key=lambda x: x['final_score'],
        reverse=True
    )

    # Return top_k results
    top_results = sorted_results[:top_k]
    # Trước khi trả về kết quả, chuyển đổi đường dẫn
    final_results = []
    for r in top_results:
        # Trích xuất tên file từ đường dẫn
        filename = os.path.basename(r['file_name'])

        # Tạo dictionary kết quả với đường dẫn mới
        final_results.append({
            'image_id': r['image_id'],
            'file_name': r['file_name'],
            'filename': filename,  # Thêm tên file ngắn
            'path': f'/static/images/{filename}',  # Đường dẫn cho Flask
            'similarity': r['final_score']
        })

    return final_results


def get_feature_stats():
    """Get statistics about stored features."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()

        # Query to count features by type
        cursor.execute(
            """
            SELECT feature_type, COUNT(*)
            FROM features
            GROUP BY feature_type
            """
        )

        results = cursor.fetchall()

        cursor.close()
        conn.close()

        return {
            feature_type: count
            for feature_type, count in results
        }

    except Exception as e:
        print(f"Error getting feature stats: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    query_image = "../data/test_image.jpg"  # Path to test image

    # Print feature statistics
    stats = get_feature_stats()
    if stats:
        print("Feature statistics:")
        for feature_type, count in stats.items():
            print(f"  {feature_type}: {count}")

    # Custom weights for ensemble search
    weights = {
        'mediapipe_landmarks': 0.5,
        'mediapipe_detection': 0.2,
        'facial_attributes': 0.2,
        'hog': 0.1
    }

    results = ensemble_search(query_image, weights=weights)

    if results:
        print("\nTop similar images:")
        for i, result in enumerate(results):
            print(f"{i + 1}. {result['file_name']} (Similarity: {result['similarity']:.4f})")
    else:
        print("No similar images found")