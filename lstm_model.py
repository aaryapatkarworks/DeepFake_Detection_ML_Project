"""
LSTM Model for Deepfake Detection
Handles temporal sequences for video deepfake detection
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, TimeDistributed, Input
from tensorflow.keras.optimizers import Adam
import numpy as np

class LSTMModel:
    def __init__(self, sequence_length=20, feature_dim=2048, num_classes=2):
        """
        Initialize LSTM Model for Deepfake Detection

        Args:
            sequence_length: Number of frames in sequence
            feature_dim: Dimension of extracted features from each frame
            num_classes: Number of classes (2 for real/fake)
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.model = None

    def create_lstm_model(self):
        """Create LSTM model for temporal sequence analysis"""
        model = Sequential([
            Input(shape=(self.sequence_length, self.feature_dim)),

            # First LSTM layer with return sequences
            LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            BatchNormalization(),

            # Second LSTM layer with return sequences
            LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            BatchNormalization(),

            # Final LSTM layer without return sequences
            LSTM(64, dropout=0.3, recurrent_dropout=0.3),
            BatchNormalization(),

            # Dense layers
            Dense(128, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),

            Dense(64, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),

            Dense(self.num_classes, activation='softmax')
        ])

        return model

    def create_bidirectional_lstm_model(self):
        """Create Bidirectional LSTM model for better temporal understanding"""
        from tensorflow.keras.layers import Bidirectional

        model = Sequential([
            Input(shape=(self.sequence_length, self.feature_dim)),

            # Bidirectional LSTM layers
            Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
            BatchNormalization(),

            Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
            BatchNormalization(),

            Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
            BatchNormalization(),

            # Dense layers
            Dense(128, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),

            Dense(64, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),

            Dense(self.num_classes, activation='softmax')
        ])

        return model

    def create_attention_lstm_model(self):
        """Create LSTM model with attention mechanism"""
        from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

        # Input layer
        inputs = Input(shape=(self.sequence_length, self.feature_dim))

        # LSTM layers
        lstm_out = LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(inputs)
        lstm_out = BatchNormalization()(lstm_out)

        # Multi-head attention
        attention_out = MultiHeadAttention(
            num_heads=8,
            key_dim=32,
            dropout=0.3
        )(lstm_out, lstm_out)

        # Add & Norm
        attention_out = LayerNormalization()(attention_out + lstm_out)

        # Final LSTM layer
        lstm_final = LSTM(128, dropout=0.3, recurrent_dropout=0.3)(attention_out)
        lstm_final = BatchNormalization()(lstm_final)

        # Dense layers
        dense_out = Dense(128, activation='relu')(lstm_final)
        dense_out = Dropout(0.5)(dense_out)
        dense_out = BatchNormalization()(dense_out)

        dense_out = Dense(64, activation='relu')(dense_out)
        dense_out = Dropout(0.5)(dense_out)
        dense_out = BatchNormalization()(dense_out)

        outputs = Dense(self.num_classes, activation='softmax')(dense_out)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def build_model(self, model_type='lstm'):
        """Build the LSTM model"""
        if model_type == 'lstm':
            self.model = self.create_lstm_model()
        elif model_type == 'bidirectional':
            self.model = self.create_bidirectional_lstm_model()
        elif model_type == 'attention':
            self.model = self.create_attention_lstm_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return self.model

    def train(self, train_data, validation_data, epochs=100, batch_size=32):
        """Train the LSTM model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='best_lstm_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]

        # Train the model
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict(self, sequences):
        """Make predictions on sequences"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        predictions = self.model.predict(sequences)
        return predictions

    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        self.model.save(filepath)

    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        return self.model

    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        return self.model.summary()

# Example usage
if __name__ == "__main__":
    # Create LSTM model
    lstm = LSTMModel()
    model = lstm.build_model(model_type='bidirectional')
    print("LSTM Model created successfully!")
    print(lstm.get_model_summary())
