#!/usr/bin/env python3
"""train_model.py: Trains a neural network model using collected driving data 
to enable autonomous navigation of the RC car."""

import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
import tensorflow as tf
import time
import datetime
from image_preprocessing import ImagePreprocessor, reshape_for_model

# Initialize our image preprocessor
preprocessor = ImagePreprocessor(image_height=240, image_width=320)

def load_training_data(data_dir='data_set'):
    """Load training data from NPZ files in the specified directory."""
    X = np.empty((0, 38400))  # Image array (flattened 320x120 images)
    y = np.empty((0, 4))      # Labels array (4 possible directions)
    
    print("Loading training data...")
    # Get all .npz files in the data_set directory
    data_files = glob.glob(os.path.join(data_dir, '*.npz'))
    
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}. Run collect_data.py first.")
    
    # Load each data file and concatenate the data
    for file in data_files:
        print(f"Loading {file}...")
        data = np.load(file)
        
        # Check data integrity and print shapes
        print(f"  Images shape: {data['images'].shape}")
        print(f"  Labels shape: {data['labels'].shape}")
        
        X = np.vstack((X, data['images']))
        y = np.vstack((y, data['labels']))
    
    print(f"Total samples loaded: {X.shape[0]}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Check for missing classes
    present_classes = np.unique(np.argmax(y, axis=1))
    class_mapping = ['Left', 'Right', 'Forward', 'Reverse']
    present_class_names = [class_mapping[i] for i in present_classes if i < len(class_mapping)]
    
    print(f"Classes present in the data: {present_classes}")
    print(f"Class names: {present_class_names}")
    
    # Check for missing classes
    all_classes = set(range(y.shape[1]))
    missing_classes = all_classes - set(present_classes)
    if missing_classes:
        missing_class_names = [class_mapping[i] for i in missing_classes if i < len(class_mapping)]
        print(f"WARNING: Missing classes in the data: {missing_classes}")
        print(f"Missing class names: {missing_class_names}")
        print("Consider collecting more data for these classes to improve model performance.")
    
    return X, y

def preprocess_data(X, images_per_batch=100):
    """Apply our custom preprocessing directly to the ROI images."""
    print("Preprocessing training data...")
    
    # Reshape to get actual ROI images (from flattened arrays)
    images = X.reshape(-1, 120, 320)
    
    # Process in batches to avoid memory issues
    processed_data = []
    total_batches = len(images) // images_per_batch + (1 if len(images) % images_per_batch > 0 else 0)
    
    for i in range(total_batches):
        start_idx = i * images_per_batch
        end_idx = min(start_idx + images_per_batch, len(images))
        
        print(f"Processing batch {i+1}/{total_batches} (images {start_idx}-{end_idx})...")
        
        # Apply direct ROI preprocessing (rotate and enhance contrast)
        batch_processed = preprocessor.preprocess_roi_batch(images[start_idx:end_idx])
        
        processed_data.append(batch_processed)
    
    # Combine all batches
    processed_data = np.vstack(processed_data)
    
    # Reshape for CNN input
    processed_data = reshape_for_model(processed_data, height=120, width=320)
    
    return processed_data

def build_model(input_shape=(120, 320, 1)):
    """Build a CNN model for self-driving car."""
    model = Sequential([
        # First convolutional layer
        Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional layer
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        
        # Third convolutional layer
        Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        
        # Fourth convolutional layer
        Conv2D(64, (3, 3), activation='relu'),
        
        # Fifth convolutional layer
        Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten the output
        Flatten(),
        
        # Fully connected layers
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='relu'),
        
        # Output layer (4 classes: left, right, forward, reverse)
        Dense(4, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Higher initial learning rate with scheduler
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X, y, model_save_path='models'):
    """Train the model and save it."""
    # Create directories if they don't exist
    os.makedirs(model_save_path, exist_ok=True)
    log_dir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data augmentation (if needed, uncomment and implement)
    # train_datagen = ImageDataGenerator(
    #     rotation_range=10,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     brightness_range=[0.8, 1.2],
    #     zoom_range=0.1,
    # )
    
    # Build the model
    model = build_model()
    model.summary()
    
    # Callbacks for training
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(model_save_path, 'model_best.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=30,  # Increased patience from 20 to 30
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=400,  # Increased from 200 to 400
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save(os.path.join(model_save_path, 'model_final.h5'))
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """Plot the training and validation accuracy and loss."""
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    scores = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {scores[0]}")
    print(f"Test Accuracy: {scores[1]}")
    
    # Calculate per-class accuracy
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Print confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Determine actual classes present in the data
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = len(unique_classes)
    
    # Create appropriate class names based on actual classes
    class_mapping = ['Left', 'Right', 'Forward', 'Reverse']
    class_names = [class_mapping[i] for i in unique_classes if i < len(class_mapping)]
    
    # Print classification report with only the classes present in the data
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    # Print warning if number of classes doesn't match expected
    if num_classes != 4:
        print(f"WARNING: Expected 4 classes but found {num_classes} classes in the data.")
        print(f"Classes found: {', '.join([class_mapping[i] for i in unique_classes if i < len(class_mapping)])}")
        missing_classes = [class_mapping[i] for i in range(len(class_mapping)) if i not in unique_classes and i < len(class_mapping)]
        if missing_classes:
            print(f"Missing classes: {', '.join(missing_classes)}")

def main():
    # Fix for TensorFlow memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for dev in physical_devices:
                tf.config.experimental.set_memory_growth(dev, True)
            print(f"Using GPU: {physical_devices}")
        except:
            print("Failed to set memory growth for GPU")
    
    # Use proper paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data_set')
    model_save_path = os.path.join(script_dir, 'models')
    
    print(f"Looking for data in: {data_dir}")
    
    # Load and preprocess data
    try:
        X, y = load_training_data(data_dir=data_dir)
        X = preprocess_data(X)
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        
        # Train model
        model, history = train_model(X_train, y_train, model_save_path=model_save_path)
        
        # Evaluate model
        evaluate_model(model, X_test, y_test)
        
        print("Model training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main() 