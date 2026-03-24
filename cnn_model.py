"""
CNN Model for Deepfake Detection
Based on research findings and best practices for deepfake detection
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0, ResNet50, VGG16
from tensorflow.keras.optimizers import Adam
import numpy as np

class CNNModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Initialize CNN Model for Deepfake Detection

        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of classes (2 for real/fake)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def create_custom_cnn(self):
        """Create a custom CNN architecture"""
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Fully Connected Layers
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),

            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),

            Dense(self.num_classes, activation='softmax')
        ])

        return model

    def create_efficientnet_model(self):
        """Create CNN model using EfficientNetB0 as backbone"""
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze base model layers
        base_model.trainable = False

        # Add custom top layers
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

        return model

    def create_resnet_model(self):
        """Create CNN model using ResNet50 as backbone"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze base model layers
        base_model.trainable = False

        # Add custom top layers
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

        return model

    def build_model(self, model_type='efficientnet'):
        """Build the CNN model"""
        if model_type == 'custom':
            self.model = self.create_custom_cnn()
        elif model_type == 'efficientnet':
            self.model = self.create_efficientnet_model()
        elif model_type == 'resnet':
            self.model = self.create_resnet_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return self.model

    def train(self, train_data, validation_data, epochs=50, batch_size=32):
        """Train the CNN model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='best_cnn_model.h5',
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

    def predict(self, images):
        """Make predictions on images"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        predictions = self.model.predict(images)
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
    # Create CNN model
    cnn = CNNModel()
    model = cnn.build_model(model_type='efficientnet')
    print("CNN Model created successfully!")
    print(cnn.get_model_summary())
