# app/app.py
import os
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import sys
import time

# Thêm đường dẫn đến thư mục scripts vào sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from scripts.search_engine import ensemble_search, get_feature_stats
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
VIZ_FOLDER = 'static/visualizations'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIZ_FOLDER'] = VIZ_FOLDER

# Tạo thư mục uploads và visualizations nếu chưa tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIZ_FOLDER'], exist_ok=True)

# Khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def visualize_face_mesh(image_path, output_path):
    """Visualize MediaPipe Face Mesh landmarks on image."""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return False

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:

            # Process image
            results = face_mesh.process(image_rgb)

            if not results.multi_face_landmarks:
                return False

            # Draw landmarks
            annotated_image = image.copy()

            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # Draw contours
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

                # Draw irises if available
                if results.multi_face_landmarks and getattr(mp_face_mesh, 'FACEMESH_IRISES', None):
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )

            # Save annotated image
            cv2.imwrite(output_path, annotated_image)
            return True

    except Exception as e:
        print(f"Error visualizing face mesh: {e}")
        return False


def visualize_similarity_chart(result_list, output_path):
    """Create a bar chart showing similarity scores."""
    try:
        # Extract filenames and similarity scores
        filenames = [os.path.basename(r['file_name']) for r in result_list]
        similarities = [r['similarity'] * 100 for r in result_list]  # Convert to percentage

        # Create the plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(filenames)), similarities, color='skyblue')

        # Add data labels
        for bar, similarity in zip(bars, similarities):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{similarity:.2f}%",
                ha='center'
            )

        # Customize the plot
        plt.title('Độ tương đồng của top ảnh so với ảnh đầu vào')
        plt.xlabel('Ảnh')
        plt.ylabel('Độ tương đồng (%)')
        plt.xticks(range(len(filenames)), filenames, rotation=45)
        plt.ylim(0, max(similarities) * 1.2)  # Add space for data labels
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_path)
        plt.close()
        return True

    except Exception as e:
        print(f"Error creating similarity chart: {e}")
        return False


@app.route('/')
def home():
    # Get feature statistics
    stats = get_feature_stats()
    return render_template('index.html', feature_stats=stats)


@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        flash('Không tìm thấy file')
        return redirect(url_for('home'))

    file = request.files['file']

    if file.filename == '':
        flash('Chưa chọn file')
        return redirect(url_for('home'))

    if file and allowed_file(file.filename):
        # Save query image
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Generate a timestamp ID for visualization files
        timestamp = int(time.time())

        # Create visualization of face mesh
        viz_filename = f"facemesh_{timestamp}.jpg"
        viz_path = os.path.join(app.config['VIZ_FOLDER'], viz_filename)
        face_viz_success = visualize_face_mesh(file_path, viz_path)

        # If face not detected, show error
        if not face_viz_success:
            flash('Không phát hiện khuôn mặt trong ảnh. Vui lòng thử ảnh khác.')
            return redirect(url_for('home'))

        # Get feature weights from form
        weights = {
            'mediapipe_landmarks': float(request.form.get('weight_landmarks', 0.4)),
            'mediapipe_detection': float(request.form.get('weight_detection', 0.2)),
            'facial_attributes': float(request.form.get('weight_attributes', 0.2)),
            'hog': float(request.form.get('weight_hog', 0.2))
        }

        # Normalize weights to ensure they sum to 1
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            for key in weights:
                weights[key] /= weight_sum

        # Find similar images with custom weights
        start_time = time.time()
        results = ensemble_search(file_path, top_k=3, weights=weights)
        search_time = time.time() - start_time

        if results:
            # Create visualization of similarity scores
            chart_filename = f"similarity_chart_{timestamp}.png"
            chart_path = os.path.join(app.config['VIZ_FOLDER'], chart_filename)
            visualize_similarity_chart(results, chart_path)

            return render_template('results.html',
                                   query_image=f"uploads/{filename.replace(os.sep, '/')}",
                                   face_viz=f"visualizations/{viz_filename.replace(os.sep, '/')}",
                                   similarity_chart = f"visualizations/{chart_filename.replace(os.sep, '/')}",
                                   results=results,
                                   search_time=search_time,
                                   weights=weights)
        else:
            flash('Không tìm thấy ảnh tương tự hoặc có lỗi xảy ra')
            return redirect(url_for('home'))
    else:
        flash('Loại file không hợp lệ. Vui lòng tải lên ảnh (jpg, jpeg, png, gif)')
        return redirect(url_for('home'))


@app.route('/compare/<path:img1>/<path:img2>')
def compare_images(img1, img2):
    """Compare two images side by side with feature visualizations."""
    # Generate face mesh visualizations for both images
    timestamp = int(time.time())

    # Ensure paths are relative to the app folder
    img1_path = os.path.abspath(os.path.join('static', img1))
    img2_path = os.path.abspath(os.path.join('static', img2))

    # Create visualizations
    viz1_filename = f"compare1_{timestamp}.jpg"
    viz2_filename = f"compare2_{timestamp}.jpg"

    viz1_path = os.path.join(app.config['VIZ_FOLDER'], viz1_filename)
    viz2_path = os.path.join(app.config['VIZ_FOLDER'], viz2_filename)

    visualize_face_mesh(img1_path, viz1_path)
    visualize_face_mesh(img2_path, viz2_path)

    return render_template('compare.html',
                           img1=img1,
                           img2=img2,
                           viz1=os.path.join('visualizations', viz1_filename),
                           viz2=os.path.join('visualizations', viz2_filename))

if __name__ == '__main__':
    app.run(debug=True)