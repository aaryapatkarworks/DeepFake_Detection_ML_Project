"""
Flask Web Application for Deepfake Detection
Provides web interface for uploading and analyzing images/videos
"""

from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import json
import base64
from datetime import datetime
import logging

# Import our models
from ensemble_model import DeepfakeEnsemble
from data_preprocessing import DataPreprocessor

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024  # 150MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models and preprocessor
ensemble_model = None
preprocessor = DataPreprocessor()

# Class index mapping
# After alignment, models emit probabilities in the order: ['real', 'fake']
# Set FAKE_CLASS_INDEX accordingly. You can override via env var FAKE_CLASS_INDEX.
FAKE_CLASS_INDEX = int(os.getenv('FAKE_CLASS_INDEX', '1'))
# Decision threshold for classifying as fake (0-1). You can override via env var FAKE_CONFIDENCE_THRESHOLD.
FAKE_CONFIDENCE_THRESHOLD = float(os.getenv('FAKE_CONFIDENCE_THRESHOLD', '0.7'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load pre-trained models"""
    global ensemble_model
    try:
        # Initialize ensemble model
        ensemble_model = DeepfakeEnsemble()

        # Try to load pre-trained models using absolute paths from this file's directory
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_paths = {
            'cnn': os.path.join(BASE_DIR, 'best_cnn_model.h5'),
            'lstm': os.path.join(BASE_DIR, 'best_lstm_model.h5'),
            'xception': os.path.join(BASE_DIR, 'best_xception_model.h5')
        }

        # Check if models exist
        models_exist = all(os.path.exists(path) for path in model_paths.values())

        if models_exist:
            ensemble_model.load_trained_models(
                model_paths['cnn'],
                model_paths['lstm'],
                model_paths['xception']
            )
            logger.info("All models loaded successfully!")
        else:
            logger.warning("Pre-trained models not found. Please train models first.")
            ensemble_model = None

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        ensemble_model = None

def process_image(image_path):
    """Process uploaded image and make prediction"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess image (match training distribution): no face-cropping, no normalization here
        # Normalization is applied inside the ensemble model paths
        processed_image = preprocessor.preprocess_image(image_rgb, detect_face=False, normalize=False)

        if ensemble_model is None:
            # Return demo prediction if models not loaded
            return {
                'prediction': 'fake',
                'confidence': 0.85,
                'individual_predictions': {
                    'cnn': {'prediction': 'fake', 'confidence': 0.82},
                    'lstm': {'prediction': 'fake', 'confidence': 0.88},
                    'xception': {'prediction': 'fake', 'confidence': 0.85}
                },
                'message': 'Demo mode - models not loaded'
            }

        # Make prediction using ensemble
        ensemble_pred = ensemble_model.predict_image(processed_image)

        # Convert prediction to readable format
        fake_confidence = float(ensemble_pred[0][FAKE_CLASS_INDEX])
        logger.info(f"Ensemble probabilities [real,fake]: {ensemble_pred[0]}, fake_index={FAKE_CLASS_INDEX}, threshold={FAKE_CONFIDENCE_THRESHOLD}")
        prediction = 'fake' if fake_confidence > FAKE_CONFIDENCE_THRESHOLD else 'real'
        confidence = fake_confidence if prediction == 'fake' else 1 - fake_confidence

        return {
            'prediction': prediction,
            'confidence': confidence,
            'individual_predictions': {
                'ensemble': {'prediction': prediction, 'confidence': confidence}
            }
        }

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {
            'error': str(e),
            'prediction': 'unknown',
            'confidence': 0.0
        }

def process_video(video_path):
    """Process uploaded video and make prediction"""
    try:
        # Extract frames from video
        frames = preprocessor.extract_frames_from_video(video_path, max_frames=50)

        if len(frames) == 0:
            raise ValueError("No frames extracted from video")

        if ensemble_model is None:
            # Return demo prediction if models not loaded
            return {
                'prediction': 'fake',
                'confidence': 0.78,
                'frames_analyzed': len(frames),
                'individual_predictions': {
                    'cnn': {'prediction': 'fake', 'confidence': 0.75},
                    'lstm': {'prediction': 'fake', 'confidence': 0.80},
                    'xception': {'prediction': 'fake', 'confidence': 0.79}
                },
                'message': 'Demo mode - models not loaded'
            }

        # Make prediction using ensemble
        ensemble_pred, individual_preds = ensemble_model.predict_video_frames(frames)

        # Convert prediction to readable format
        fake_confidence = float(ensemble_pred[FAKE_CLASS_INDEX])
        logger.info(f"Video ensemble probabilities [real,fake]: {ensemble_pred}, fake_index={FAKE_CLASS_INDEX}, threshold={FAKE_CONFIDENCE_THRESHOLD}")
        prediction = 'fake' if fake_confidence > FAKE_CONFIDENCE_THRESHOLD else 'real'
        confidence = fake_confidence if prediction == 'fake' else 1 - fake_confidence

        return {
            'prediction': prediction,
            'confidence': confidence,
            'frames_analyzed': len(frames),
            'individual_predictions': individual_preds
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {
            'error': str(e),
            'prediction': 'unknown',
            'confidence': 0.0,
            'frames_analyzed': 0
        }

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    """Upload page"""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Determine file type and process accordingly
            file_extension = filename.rsplit('.', 1)[1].lower()

            if file_extension in ['png', 'jpg', 'jpeg', 'gif']:
                result = process_image(filepath)
                file_type = 'image'
            else:
                result = process_video(filepath)
                file_type = 'video'

            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass

            return render_template('result.html', 
                                 result=result, 
                                 file_type=file_type,
                                 filename=file.filename)

        except Exception as e:
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass

            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('upload_page'))
    else:
        flash('Invalid file type. Please upload an image or video file.')
        return redirect(request.url)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for file analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Save uploaded file temporarily
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Process file
        file_extension = filename.rsplit('.', 1)[1].lower()

        if file_extension in ['png', 'jpg', 'jpeg', 'gif']:
            result = process_image(filepath)
        else:
            result = process_video(filepath)

        # Clean up
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify(result)

    except Exception as e:
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/api/model_status')
def model_status():
    """API endpoint to check model status"""
    status = {
        'models_loaded': ensemble_model is not None,
        'available_models': ['CNN', 'LSTM', 'Xception', 'Ensemble'],
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'fake_class_index': FAKE_CLASS_INDEX,
        'fake_confidence_threshold': FAKE_CONFIDENCE_THRESHOLD
    }
    return jsonify(status)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return "File is too large. Maximum size is 150MB.", 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Load models on startup
    load_models()

    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)
