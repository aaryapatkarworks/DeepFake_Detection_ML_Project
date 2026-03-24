"""
Multi-Model Training Script for Deepfake Detection
Trains CNN, LSTM, and Xception models efficiently
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, LSTM, TimeDistributed
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import cv2
from pathlib import Path

def create_cnn_model():
    """Create CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_lstm_model():
    """Create LSTM model for sequence data"""
    model = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(20, 224, 224, 3)),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Dropout(0.25)),
        
        TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Dropout(0.25)),
        
        TimeDistributed(Flatten()),
        LSTM(128, dropout=0.3, recurrent_dropout=0.3),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_xception_model():
    """Create Xception-based model"""
    from tensorflow.keras.applications import Xception
    from tensorflow.keras.layers import GlobalAveragePooling2D
    
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3)
    )
    
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_lstm_data(data_dir, batch_size=32):
    """Prepare sequence data for LSTM"""
    print("Preparing LSTM sequence data...")
    
    sequences = []
    labels = []
    
    for class_name in ['real', 'fake']:
        class_dir = Path(data_dir) / class_name
        images = list(class_dir.glob('*.jpg'))
        
        # Group images into sequences of 20
        for i in range(0, len(images) - 19, 20):
            sequence_images = images[i:i+20]
            sequence = []
            
            for img_path in sequence_images:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32) / 255.0
                sequence.append(img)
            
            sequences.append(np.array(sequence))
            labels.append(0 if class_name == 'real' else 1)
    
    return np.array(sequences), np.array(labels)

def train_cnn_model():
    """Train CNN model"""
    print("=" * 50)
    print("TRAINING CNN MODEL")
    print("=" * 50)
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        'data/val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    # Create and train model
    model = create_cnn_model()
    
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint('best_cnn_model.h5', save_best_only=True, monitor='val_accuracy')
    ]
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )
    
    print("CNN Model trained and saved!")
    return model

def train_lstm_model():
    """Train LSTM model"""
    print("=" * 50)
    print("TRAINING LSTM MODEL")
    print("=" * 50)
    
    # Prepare sequence data
    X_train, y_train = prepare_lstm_data('data/train')
    X_val, y_val = prepare_lstm_data('data/val')
    
    # Convert labels to categorical
    from tensorflow.keras.utils import to_categorical
    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)
    
    # Create and train model
    model = create_lstm_model()
    
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint('best_lstm_model.h5', save_best_only=True, monitor='val_accuracy')
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    print("LSTM Model trained and saved!")
    return model

def train_xception_model():
    """Train Xception model"""
    print("=" * 50)
    print("TRAINING XCEPTION MODEL")
    print("=" * 50)
    
    # Data generators for Xception (299x299)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(299, 299),
        batch_size=16,  # Smaller batch for Xception
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        'data/val',
        target_size=(299, 299),
        batch_size=16,
        class_mode='categorical'
    )
    
    # Create and train model
    model = create_xception_model()
    
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint('best_xception_model.h5', save_best_only=True, monitor='val_accuracy')
    ]
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=8,  # Fewer epochs for Xception
        callbacks=callbacks,
        verbose=1
    )
    
    print("Xception Model trained and saved!")
    return model

def main():
    """Main training function"""
    print("Starting Multi-Model Training for Deepfake Detection")
    print("This will train CNN, LSTM, and Xception models sequentially")
    print("Estimated time: 30-60 minutes")
    print()
    
    try:
        # Train models sequentially
        cnn_model = train_cnn_model()
        print()
        
        lstm_model = train_lstm_model()
        print()
        
        xception_model = train_xception_model()
        print()
        
        print("=" * 60)
        print("ALL MODELS TRAINED SUCCESSFULLY!")
        print("=" * 60)
        print("Saved models:")
        print("- best_cnn_model.h5")
        print("- best_lstm_model.h5") 
        print("- best_xception_model.h5")
        print()
        print("You can now run: python flask_app.py")
        print("The Flask app will load all trained models!")
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
