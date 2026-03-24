"""
Ensemble Model for Deepfake Detection
Combines CNN, LSTM, and Xception models for improved performance
"""

import os
import sys
import tensorflow as tf
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import joblib

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class EnsembleModel:
    def __init__(self, models=None, weights=None):
        """
        Initialize Ensemble Model

        Args:
            models: List of trained models [cnn_model, lstm_model, xception_model]
            weights: Weights for each model in ensemble voting
        """
        self.models = models or []
        self.weights = weights or [1/len(self.models) if self.models else 0] * len(self.models)
        self.meta_learner = None
        self.ensemble_type = 'voting'  # 'voting', 'stacking', or 'weighted_average'

    def add_model(self, model, weight=1.0):
        """Add a model to the ensemble"""
        self.models.append(model)
        self.weights.append(weight)
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w/total_weight for w in self.weights]

    def voting_ensemble_predict(self, X):
        """Predict using voting ensemble"""
        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Convert to numpy array for easier manipulation
        predictions = np.array(predictions)

        # Hard voting (majority vote)
        final_predictions = np.mean(predictions, axis=0)

        return final_predictions

    def weighted_average_predict(self, X):
        """Predict using weighted average ensemble"""
        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            weighted_pred = pred * self.weights[i]
            predictions.append(weighted_pred)

        # Sum weighted predictions
        final_predictions = np.sum(predictions, axis=0)

        return final_predictions

    def prepare_stacking_data(self, X_train, y_train, X_val, y_val):
        """Prepare data for stacking ensemble"""
        if not self.models:
            raise ValueError("No models in ensemble")

        # Get predictions from base models on training data
        train_predictions = []
        val_predictions = []

        for model in self.models:
            # Training predictions (using cross-validation would be better)
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_predictions.append(train_pred)
            val_predictions.append(val_pred)

        # Stack predictions horizontally
        X_train_meta = np.hstack(train_predictions)
        X_val_meta = np.hstack(val_predictions)

        return X_train_meta, X_val_meta

    def train_stacking_ensemble(self, X_train, y_train, X_val, y_val, meta_learner=None):
        """Train stacking ensemble with meta-learner"""
        # Prepare meta-features
        X_train_meta, X_val_meta = self.prepare_stacking_data(X_train, y_train, X_val, y_val)

        # Train meta-learner
        if meta_learner is None:
            meta_learner = LogisticRegression(random_state=42, max_iter=1000)

        # Convert one-hot encoded labels to single labels if needed
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train_single = np.argmax(y_train, axis=1)
            y_val_single = np.argmax(y_val, axis=1)
        else:
            y_train_single = y_train
            y_val_single = y_val

        self.meta_learner = meta_learner
        self.meta_learner.fit(X_train_meta, y_train_single)

        # Evaluate on validation set
        val_pred = self.meta_learner.predict(X_val_meta)
        accuracy = accuracy_score(y_val_single, val_pred)

        print(f"Stacking ensemble validation accuracy: {accuracy:.4f}")
        return self.meta_learner

    def stacking_predict(self, X):
        """Predict using stacking ensemble"""
        if not self.models or self.meta_learner is None:
            raise ValueError("Stacking ensemble not trained")

        # Get base model predictions
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Stack predictions
        X_meta = np.hstack(predictions)

        # Get final prediction from meta-learner
        final_pred = self.meta_learner.predict(X_meta)
        final_proba = self.meta_learner.predict_proba(X_meta)

        return final_pred, final_proba

    def predict(self, X, ensemble_type=None):
        """Make ensemble predictions"""
        if ensemble_type is None:
            ensemble_type = self.ensemble_type

        if ensemble_type == 'voting':
            return self.voting_ensemble_predict(X)
        elif ensemble_type == 'weighted_average':
            return self.weighted_average_predict(X)
        elif ensemble_type == 'stacking':
            pred, proba = self.stacking_predict(X)
            return proba
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    def evaluate_ensemble(self, X_test, y_test, ensemble_types=['voting', 'weighted_average']):
        """Evaluate different ensemble methods"""
        results = {}

        for ens_type in ensemble_types:
            if ens_type == 'stacking' and self.meta_learner is None:
                print(f"Skipping {ens_type} - meta-learner not trained")
                continue

            predictions = self.predict(X_test, ensemble_type=ens_type)

            # Convert predictions to class labels
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                pred_labels = np.argmax(predictions, axis=1)
            else:
                pred_labels = (predictions > 0.5).astype(int)

            # Convert ground truth if needed
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                true_labels = np.argmax(y_test, axis=1)
            else:
                true_labels = y_test

            # Calculate metrics
            accuracy = accuracy_score(true_labels, pred_labels)
            results[ens_type] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'pred_labels': pred_labels
            }

            print(f"{ens_type.title()} Ensemble Accuracy: {accuracy:.4f}")
            print(f"Classification Report for {ens_type}:")
            print(classification_report(true_labels, pred_labels))
            print("-" * 50)

        return results

    def save_ensemble(self, filepath):
        """Save the ensemble model"""
        ensemble_data = {
            'weights': self.weights,
            'ensemble_type': self.ensemble_type,
            'meta_learner': self.meta_learner
        }

        # Save ensemble configuration
        with open(f"{filepath}_config.pkl", 'wb') as f:
            pickle.dump(ensemble_data, f)

        # Note: Individual models should be saved separately
        print(f"Ensemble configuration saved to {filepath}_config.pkl")
        print("Note: Save individual models separately using their respective save methods")

    def load_ensemble(self, filepath, models):
        """Load ensemble configuration and set models"""
        # Load ensemble configuration
        with open(f"{filepath}_config.pkl", 'rb') as f:
            ensemble_data = pickle.load(f)

        self.weights = ensemble_data['weights']
        self.ensemble_type = ensemble_data['ensemble_type']
        self.meta_learner = ensemble_data['meta_learner']
        self.models = models

        print("Ensemble configuration loaded successfully")

    def get_model_contributions(self, X, sample_idx=0):
        """Get individual model contributions for a sample"""
        if not self.models:
            raise ValueError("No models in ensemble")

        contributions = {}
        sample = X[sample_idx:sample_idx+1]  # Get single sample

        for i, model in enumerate(self.models):
            pred = model.predict(sample)
            model_name = f"Model_{i+1}"
            contributions[model_name] = {
                'prediction': pred[0],
                'weight': self.weights[i],
                'weighted_contribution': pred[0] * self.weights[i]
            }

        return contributions

class DeepfakeEnsemble:
    """Complete deepfake detection ensemble combining CNN, LSTM, and Xception"""

    def __init__(self):
        self.cnn_model = None
        self.lstm_model = None  
        self.xception_model = None
        self.ensemble = EnsembleModel()
        self.feature_extractor = None

    def load_trained_models(self, cnn_path, lstm_path, xception_path):
        """Load pre-trained individual models"""
        # Load CNN model
        self.cnn_model = tf.keras.models.load_model(cnn_path)

        # Load LSTM model
        self.lstm_model = tf.keras.models.load_model(lstm_path)

        # Load Xception model
        self.xception_model = tf.keras.models.load_model(xception_path)

        # Add models to ensemble
        self.ensemble.add_model(self.cnn_model, weight=0.4)
        self.ensemble.add_model(self.lstm_model, weight=0.3) 
        self.ensemble.add_model(self.xception_model, weight=0.3)

        print("All models loaded successfully!")

    def predict_image(self, image):
        """Predict if a single image is deepfake"""
        # Preprocess image for CNN and Xception (resize and normalize to 0-1)
        cnn_input = tf.image.resize(image, [224, 224])
        cnn_input = tf.cast(cnn_input, tf.float32) / 255.0
        xception_input = tf.image.resize(image, [299, 299])
        xception_input = tf.cast(xception_input, tf.float32) / 255.0

        # Get predictions
        cnn_pred = self.cnn_model.predict(np.expand_dims(cnn_input, 0))
        xception_pred = self.xception_model.predict(np.expand_dims(xception_input, 0))

        # Map CNN and Xception from [fake, real] -> [real, fake] to align with LSTM/order
        if cnn_pred.shape[-1] == 2:
            cnn_pred = cnn_pred[:, [1, 0]]
        if xception_pred.shape[-1] == 2:
            xception_pred = xception_pred[:, [1, 0]]

        # For LSTM, we need sequence data - use repeated frame
        lstm_pred = None
        try:
            lstm_input = np.repeat(np.expand_dims(cnn_input, 0), 20, axis=0)
            lstm_input = np.expand_dims(lstm_input, 0)  # Add batch dimension
            lstm_pred = self.lstm_model.predict(lstm_input)
        except Exception:
            lstm_pred = None

        # Ensemble prediction
        if lstm_pred is not None:
            ensemble_pred = (cnn_pred * 0.4 + lstm_pred * 0.3 + xception_pred * 0.3)
        else:
            ensemble_pred = (cnn_pred * 0.5 + xception_pred * 0.5)

        return ensemble_pred

    def predict_video_frames(self, frames):
        """Predict deepfake for video frames"""
        if len(frames) == 0:
            raise ValueError("No frames provided")

        # Process frames for different models
        predictions = []

        # CNN predictions on individual frames
        cnn_preds = []
        for frame in frames:
            frame_resized = tf.image.resize(frame, [224, 224])
            frame_resized = tf.cast(frame_resized, tf.float32) / 255.0
            cnn_pred = self.cnn_model.predict(np.expand_dims(frame_resized, 0))
            cnn_preds.append(cnn_pred[0])

        # Average CNN predictions
        avg_cnn_pred = np.mean(cnn_preds, axis=0)

        # LSTM prediction on sequence (best-effort)
        lstm_pred = None
        try:
            lstm_frames = [tf.image.resize(frame, [224, 224]) for frame in frames[:20]]
            lstm_frames = [tf.cast(f, tf.float32) / 255.0 for f in lstm_frames]
            if len(lstm_frames) < 20:
                # Repeat last frame if not enough frames
                last_frame = lstm_frames[-1]
                lstm_frames.extend([last_frame] * (20 - len(lstm_frames)))

            lstm_input = np.expand_dims(np.array(lstm_frames), 0)
            lstm_pred = self.lstm_model.predict(lstm_input)[0]
        except Exception:
            lstm_pred = None

        # Xception predictions on individual frames
        xception_preds = []
        for frame in frames:
            frame_resized = tf.image.resize(frame, [299, 299])
            frame_resized = tf.cast(frame_resized, tf.float32) / 255.0
            xception_pred = self.xception_model.predict(np.expand_dims(frame_resized, 0))
            xception_preds.append(xception_pred[0])

        # Average Xception predictions
        avg_xception_pred = np.mean(xception_preds, axis=0)

        # Align CNN/Xception to [real, fake]
        if avg_cnn_pred.shape[-1] == 2:
            avg_cnn_pred = avg_cnn_pred[[1, 0]]
        if avg_xception_pred.shape[-1] == 2:
            avg_xception_pred = avg_xception_pred[[1, 0]]

        # Ensemble prediction
        if lstm_pred is not None:
            ensemble_pred = (avg_cnn_pred * 0.4 + lstm_pred * 0.3 + avg_xception_pred * 0.3)
        else:
            ensemble_pred = (avg_cnn_pred * 0.5 + avg_xception_pred * 0.5)

        return ensemble_pred, {
            'cnn_prediction': avg_cnn_pred,
            'lstm_prediction': lstm_pred if lstm_pred is not None else 'skipped',
            'xception_prediction': avg_xception_pred
        }

# Example usage
if __name__ == "__main__":
    # Create ensemble
    ensemble = DeepfakeEnsemble()

    print("Deepfake Detection Ensemble created!")
    print("Load your trained models using load_trained_models() method")
