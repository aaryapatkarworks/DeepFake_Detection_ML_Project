"""
Xception Model for Deepfake Detection
Based on research showing Xception's effectiveness in deepfake detection
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

class XceptionModel:
    def __init__(self, input_shape=(299, 299, 3), num_classes=2):
        """
        Initialize Xception Model for Deepfake Detection

        Args:
            input_shape: Shape of input images (299x299x3 for Xception)
            num_classes: Number of classes (2 for real/fake)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def create_xception_model(self, fine_tune=False):
        """Create Xception model for deepfake detection"""
        # Load pre-trained Xception model
        base_model = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze base model layers initially
        base_model.trainable = fine_tune

        # Add custom classification head
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),

            # First dense block
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),

            # Second dense block
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),

            # Third dense block
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),

            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])

        return model

    def create_custom_xception_head(self):
        """Create Xception model with custom classification head optimized for deepfakes"""
        base_model = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze base model initially
        base_model.trainable = False

        # Custom head designed for deepfake detection
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # Multi-branch architecture
        branch1 = Dense(512, activation='relu')(x)
        branch1 = BatchNormalization()(branch1)
        branch1 = Dropout(0.5)(branch1)
        branch1 = Dense(256, activation='relu')(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Dropout(0.3)(branch1)

        branch2 = Dense(256, activation='relu')(x)
        branch2 = BatchNormalization()(branch2)
        branch2 = Dropout(0.4)(branch2)
        branch2 = Dense(128, activation='relu')(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = Dropout(0.3)(branch2)

        # Concatenate branches
        from tensorflow.keras.layers import concatenate
        merged = concatenate([branch1, branch2])

        # Final layers
        merged = Dense(256, activation='relu')(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(0.5)(merged)

        predictions = Dense(self.num_classes, activation='softmax')(merged)

        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def build_model(self, model_type='standard', fine_tune=False):
        """Build the Xception model"""
        if model_type == 'standard':
            self.model = self.create_xception_model(fine_tune=fine_tune)
        elif model_type == 'custom_head':
            self.model = self.create_custom_xception_head()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Compile the model
        if fine_tune:
            # Lower learning rate for fine-tuning
            optimizer = Adam(learning_rate=0.0001)
        else:
            # Higher learning rate for training from scratch
            optimizer = Adam(learning_rate=0.001)

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return self.model

    def fine_tune_model(self, unfreeze_layers=20):
        """Fine-tune the model by unfreezing top layers"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Unfreeze the top layers of the base model
        base_model = self.model.layers[0]
        base_model.trainable = True

        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) - unfreeze_layers

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # Re-compile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        print(f"Fine-tuning enabled for top {unfreeze_layers} layers")

    def train(self, train_data, validation_data, epochs=50, batch_size=32):
        """Train the Xception model"""
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
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='best_xception_model.h5',
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
    # Create Xception model
    xception = XceptionModel()
    model = xception.build_model(model_type='custom_head')
    print("Xception Model created successfully!")
    print(xception.get_model_summary())
