#!/usr/bin/env python3
"""final_model.py: Final training script with better preprocessing and more robust model."""

import os
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import datetime
from sklearn.metrics import confusion_matrix, classification_report


def load_training_data(data_dir='../data_set'):
    """Load training data from NPZ files in the specified directory."""
    image_size = 240 * 320
    X = np.empty((0, image_size))
    y = np.empty((0, 4))
    
    print("Loading training data...")
    data_files = glob.glob(os.path.join(data_dir, '*.npz'))
    
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}. Run collect_data.py first.")
    
    for file in data_files:
        print(f"Loading {file}...")
        data = np.load(file)
        
        # Ensure images and labels exist in the data file
        if 'images' not in data or 'labels' not in data:
            print(f"WARNING: File {file} does not contain expected 'images' and 'labels' arrays. Skipping.")
            continue
            
        data_images = data['images']
        data_labels = data['labels']
        
        # Ensure arrays are not empty and shape is consistent
        if data_images.shape[0] == 0 or data_labels.shape[0] == 0:
            print(f"WARNING: Empty arrays in {file}. Skipping.")
            continue
            
        if data_images.shape[0] != data_labels.shape[0]:
            print(f"WARNING: Inconsistent shapes in {file}. Images: {data_images.shape}, Labels: {data_labels.shape}. Skipping.")
            continue
            
        # For the final model, we may want to use all data including reverse
        # But for now, we'll filter out reverse commands to simplify
        mask = np.argmax(data_labels, axis=1) != 3  # Filter out reverse (3)
        data_images = data_images[mask]
        data_labels = data_labels[mask]
        
        print(f"  Images shape: {data_images.shape}")
        print(f"  Labels shape: {data_labels.shape}")
        
        # Ensure labels are valid one-hot vectors
        if data_labels.shape[1] != 4:
            print(f"WARNING: Labels in {file} are not 4-dimensional one-hot vectors. Skipping.")
            continue
            
        # Check if any images have suspicious values
        if np.isnan(data_images).any() or np.isinf(data_images).any():
            print(f"WARNING: File {file} contains NaN or infinite values. Cleaning...")
            data_images = np.nan_to_num(data_images)
            
        X = np.vstack((X, data_images))
        y = np.vstack((y, data_labels))
    
    print(f"Total samples loaded: {X.shape[0]}")
    
    # Check for class distribution
    class_mapping = ['Left', 'Right', 'Forward', 'Reverse']
    present_classes = np.unique(np.argmax(y, axis=1))
    present_class_names = [class_mapping[i] for i in present_classes if i < len(class_mapping)]
    
    print(f"Classes present in the data: {present_class_names}")
    
    # Print class distribution
    class_counts = np.bincount(np.argmax(y, axis=1))
    for i, count in enumerate(class_counts):
        if i < len(class_mapping):
            print(f"Class {class_mapping[i]}: {count} samples")
    
    return X, y


def preprocess_data(X, batch_size=100, debug_output=True):
    """Enhanced preprocessing focusing on line detection."""
    print("Preprocessing training data...")
    
    # Reshape flattened arrays to original image dimensions
    images = X.reshape(-1, 240, 320)
    
    # Process in batches to avoid memory issues
    processed_data = []
    total_batches = len(images) // batch_size + (1 if len(images) % batch_size > 0 else 0)
    
    # Save example images for debugging
    if debug_output:
        example_dir = "preprocessing_examples"
        os.makedirs(example_dir, exist_ok=True)
    
    # Track empty images 
    empty_count = 0
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(images))
        
        print(f"Processing batch {i+1}/{total_batches} (images {start_idx}-{end_idx})...")
        
        # Process images in this batch
        batch_processed = []
        for img_idx, img in enumerate(images[start_idx:end_idx]):
            # Ensure floating point values are in the correct range
            if img.max() <= 1.0:
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                img_uint8 = img.astype(np.uint8)
            
            # Rotate the image 180 degrees so line is in front of car
            rotated = cv2.rotate(img_uint8, cv2.ROTATE_180)
            
            # Take only bottom half - where the line should be
            roi = rotated[120:240, :]
            
            # Apply preprocessing pipeline
            # 1. Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(roi, (5, 5), 0)
            
            # 2. Try several thresholding techniques and pick the best one
            # Adaptive threshold
            adaptive = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Otsu's thresholding
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Simple threshold
            _, simple = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
            
            # Count white pixels in each method to determine best approach
            adaptive_pixels = np.sum(adaptive == 255)
            otsu_pixels = np.sum(otsu == 255)
            simple_pixels = np.sum(simple == 255)
            
            # Choose the threshold with moderate number of pixels
            # Too many or too few pixels are both problematic
            if 500 < adaptive_pixels < 10000:
                binary = adaptive
            elif 500 < otsu_pixels < 10000:
                binary = otsu
            elif 500 < simple_pixels < 10000:
                binary = simple
            else:
                # If all failed, use Canny edge detection
                edges = cv2.Canny(blurred, 50, 150)
                binary = edges
            
            # Apply morphological operations to remove noise and connect lines
            kernel = np.ones((3, 3), np.uint8)
            morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours and draw only the largest ones
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            line_mask = np.zeros_like(morphed)
            
            if contours:
                # Sort contours by area, largest first
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                # Take only the largest contours
                for contour in contours[:3]:
                    cv2.drawContours(line_mask, [contour], -1, 255, -1)
            
            # Check if the mask is empty and create a fallback
            if np.sum(line_mask) == 0:
                empty_count += 1
                # Create a synthetic line based on the class
                line_mask = np.zeros_like(morphed)
                
                # Just add a vertical line in the middle as fallback
                center_x = line_mask.shape[1] // 2
                line_mask[:, center_x-5:center_x+5] = 255
            
            # Save example images for debugging
            if debug_output and i == 0 and img_idx < 5:
                example_path = os.path.join(example_dir, f"example_{img_idx}")
                cv2.imwrite(f"{example_path}_original.png", img_uint8)
                cv2.imwrite(f"{example_path}_roi.png", roi)
                cv2.imwrite(f"{example_path}_binary.png", binary)
                cv2.imwrite(f"{example_path}_morphed.png", morphed)
                cv2.imwrite(f"{example_path}_final.png", line_mask)
            
            batch_processed.append(line_mask)
            
        batch_processed = np.array(batch_processed)
        processed_data.append(batch_processed)
    
    # Combine all batches
    processed_data = np.vstack(processed_data)
    
    # Reshape for CNN input (height, width, channels)
    processed_data = processed_data.reshape(-1, 120, 320, 1)
    
    # Simple normalization to [0,1]
    processed_data = processed_data / 255.0
    
    print(f"Processed data range: [{processed_data.min()}, {processed_data.max()}]")
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Empty images fixed: {empty_count}")
    
    return processed_data


def build_model(input_shape=(120, 320, 1), num_classes=3):
    """Build a specialized CNN for line detection and direction classification."""
    # Input layer
    frame_input = Input(shape=input_shape, name='input')
    
    # First convolutional block
    x = Conv2D(16, (5, 5), strides=(2, 2), padding='same', activation='relu')(frame_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second convolutional block
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Third convolutional block
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create and compile model
    model = Model(inputs=frame_input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_balanced_dataset(X, y, target_count=400, num_classes=3):
    """Create a balanced dataset by resampling classes."""
    # Get class indices
    y_integers = np.argmax(y, axis=1)
    unique_classes = np.unique(y_integers)
    
    # Create new one-hot labels with the correct number of classes
    new_y = np.zeros((len(y), num_classes))
    
    # Map original classes to new indices
    class_map = {original: new for new, original in enumerate(sorted(unique_classes))}
    
    # Fill in one-hot labels
    for i, original_class in enumerate(y_integers):
        new_idx = class_map[original_class]
        new_y[i, new_idx] = 1
    
    # Create storage for balanced data
    X_balanced = []
    y_balanced = []
    
    # Remap class indices based on the new one-hot encoding
    new_y_integers = np.argmax(new_y, axis=1)
    
    for class_idx in range(num_classes):
        # Get samples for this class
        class_indices = np.where(new_y_integers == class_idx)[0]
        class_count = len(class_indices)
        
        if class_count == 0:
            print(f"Warning: Class {class_idx} has no samples!")
            continue
        
        # Calculate resampling parameters
        copies_needed = target_count // class_count
        remainder = target_count % class_count
        
        print(f"Class {class_idx}: {class_count} samples -> {copies_needed} copies + {remainder} extra")
        
        # Add complete copies
        for _ in range(copies_needed):
            X_balanced.append(X[class_indices])
            y_balanced.append(new_y[class_indices])
        
        # Add remainder
        if remainder > 0:
            extra_indices = np.random.choice(class_indices, remainder, replace=False)
            X_balanced.append(X[extra_indices])
            y_balanced.append(new_y[extra_indices])
    
    # Combine all data
    X_balanced = np.vstack(X_balanced)
    y_balanced = np.vstack(y_balanced)
    
    # Shuffle the data
    indices = np.arange(len(X_balanced))
    np.random.shuffle(indices)
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]
    
    print(f"Balanced data shape: {X_balanced.shape}, {y_balanced.shape}")
    print("Class distribution:", np.bincount(np.argmax(y_balanced, axis=1)))
    
    return X_balanced, y_balanced


def train_model(X_train, y_train, X_val, y_val, model_save_path='models'):
    """Train the model with balanced data and appropriate parameters."""
    # Create directories for outputs
    os.makedirs(model_save_path, exist_ok=True)
    log_dir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Determine classes present in data
    y_train_classes = np.argmax(y_train, axis=1)
    present_classes = sorted(np.unique(y_train_classes))
    num_classes = len(present_classes)
    
    # Check if we have at least 2 classes
    if num_classes < 2:
        raise ValueError(f"Need at least 2 classes to train, but only found {num_classes}")
    
    # Map class names for better interpretability
    class_mapping = ['Left', 'Right', 'Forward', 'Reverse']
    present_class_names = [class_mapping[i] for i in present_classes if i < len(class_mapping)]
    print(f"Training model for {num_classes} classes: {present_class_names}")
    
    # Create class mapping dictionary
    class_mapping_dict = {original: new for new, original in enumerate(present_classes)}
    print("Class mapping:", class_mapping_dict)
    
    # Create balanced training dataset with the correct number of output classes
    X_train_balanced, y_train_balanced = create_balanced_dataset(
        X_train, y_train, target_count=400, num_classes=num_classes
    )
    
    # Create balanced validation dataset with the correct number of output classes
    X_val_balanced, y_val_balanced = create_balanced_dataset(
        X_val, y_val, target_count=200, num_classes=num_classes
    )
    
    # Check the shapes
    print(f"Training data shapes: X={X_train_balanced.shape}, y={y_train_balanced.shape}")
    print(f"Validation data shapes: X={X_val_balanced.shape}, y={y_val_balanced.shape}")
    
    # Build the model
    model = build_model(input_shape=(120, 320, 1), num_classes=num_classes)
    model.summary()
    
    # Create a data generator with mild augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Don't flip, as this would change left/right labels
        fill_mode='nearest'
    )
    
    # Create a generator for training data
    train_generator = train_datagen.flow(
        X_train_balanced, y_train_balanced,
        batch_size=32,
        shuffle=True
    )
    
    # Calculate steps per epoch
    steps_per_epoch = len(X_train_balanced) // 32
    
    # Define callbacks for training
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(model_save_path, 'model_best.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            verbose=1,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
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
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        validation_data=(X_val_balanced, y_val_balanced),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save(os.path.join(model_save_path, 'model_final.h5'))
    
    # Save class mapping for inference
    with open(os.path.join(model_save_path, 'class_mapping.txt'), 'w') as f:
        for original_class in present_classes:
            new_class = class_mapping_dict[original_class]
            class_name = class_mapping[original_class] if original_class < len(class_mapping) else f"Unknown-{original_class}"
            f.write(f"{new_class}: {class_name} (original class {original_class})\n")
    
    # Plot and save training history
    plot_training_history(history)
    
    return model, history, num_classes, class_mapping_dict, present_classes


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


def evaluate_model(model, X_test, y_test, num_classes, present_classes):
    """Evaluate the model on test data with detailed metrics."""
    # Create a mapping from original classes to model output positions
    class_mapping = ['Left', 'Right', 'Forward', 'Reverse']
    class_mapping_dict = {original: new for new, original in enumerate(present_classes)}
    
    # Prepare test labels to match model output shape
    y_test_remapped = np.zeros((len(y_test), num_classes))
    
    # Populate remapped test labels
    for i, original_class in enumerate(np.argmax(y_test, axis=1)):
        if original_class in class_mapping_dict:
            new_class = class_mapping_dict[original_class]
            y_test_remapped[i, new_class] = 1
    
    # Evaluate the model
    print("\nEvaluating model on test data...")
    scores = model.evaluate(X_test, y_test_remapped, verbose=1)
    print(f"Test Loss: {scores[0]}")
    print(f"Test Accuracy: {scores[1] * 100:.2f}%")
    
    # Generate predictions
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test_remapped, axis=1)
    
    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Create class names for output
    inverted_map = {new: original for original, new in class_mapping_dict.items()}
    class_names = [class_mapping[inverted_map[i]] if inverted_map[i] < len(class_mapping) 
                  else f"Class-{inverted_map[i]}" for i in range(num_classes)]
    
    # Print classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    
    # Visualize some example predictions
    visualize_predictions(model, X_test, y_test_remapped, class_names, num_samples=5)
    
    return scores[1]  # Return accuracy


def visualize_predictions(model, X_test, y_test, class_names, num_samples=5):
    """Visualize model predictions on test samples."""
    # Get random indices
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, 3*num_samples))
    
    for i, idx in enumerate(indices):
        # Get the image and true label
        img = X_test[idx]
        true_label = np.argmax(y_test[idx])
        
        # Get model prediction
        pred = model.predict(img.reshape(1, *img.shape))[0]
        pred_label = np.argmax(pred)
        
        # Plot the image
        plt.subplot(num_samples, 2, i*2+1)
        plt.imshow(img[:,:,0], cmap='gray')
        plt.title(f"True: {class_names[true_label]}")
        plt.axis('off')
        
        # Plot the prediction probabilities
        plt.subplot(num_samples, 2, i*2+2)
        plt.bar(class_names, pred)
        plt.title(f"Pred: {class_names[pred_label]} ({pred[pred_label]:.2f})")
        plt.ylim(0, 1)
        
    plt.tight_layout()
    plt.savefig('prediction_samples.png')


def main():
    """Main function to run the training pipeline."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Disable GPU if it causes issues
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for dev in physical_devices:
                tf.config.experimental.set_memory_growth(dev, True)
            print(f"Using GPU: {physical_devices}")
        else:
            print("No GPU found. Using CPU.")
    except:
        print("Failed to configure GPU. Using CPU.")
    
    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data_set')
    model_save_path = os.path.join(script_dir, 'models')
    
    print(f"Looking for data in: {data_dir}")
    print(f"Models will be saved to: {model_save_path}")
    
    try:
        # Load data
        X, y = load_training_data(data_dir=data_dir)
        
        # Check data validity
        if X.shape[0] == 0 or y.shape[0] == 0:
            print("ERROR: Empty dataset. Please check your data_set directory.")
            return
        
        # Print dataset statistics
        y_classes = np.argmax(y, axis=1)
        unique_classes = np.unique(y_classes)
        class_mapping = ['Left', 'Right', 'Forward', 'Reverse']
        
        print("\nDataset Statistics:")
        print(f"Total samples: {X.shape[0]}")
        print(f"Classes found: {[class_mapping[i] for i in unique_classes if i < len(class_mapping)]}")
        
        print("\nClass Distribution:")
        for cls in unique_classes:
            count = np.sum(y_classes == cls)
            percent = count / len(y_classes) * 100
            if cls < len(class_mapping):
                print(f"  Class {class_mapping[cls]}: {count} samples ({percent:.2f}%)")
        
        # Only proceed if we have at least 2 classes
        if len(unique_classes) < 2:
            print("ERROR: Need at least 2 classes with samples to train a classifier")
            return
        
        # Preprocess data
        X_processed = preprocess_data(X, batch_size=100)
        
        # Split data into training, validation, and test sets with stratification
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_processed, y, test_size=0.3, random_state=42, stratify=y_classes
        )
        
        # Further split the temp set into validation and test sets
        y_temp_classes = np.argmax(y_temp, axis=1)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp_classes
        )
        
        # Print split statistics
        print("\nData Split Statistics:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train the model
        model, history, num_classes, class_mapping_dict, present_classes = train_model(
            X_train, y_train, X_val, y_val, model_save_path=model_save_path
        )
        
        # Evaluate the model
        test_accuracy = evaluate_model(model, X_test, y_test, num_classes, present_classes)
        
        print(f"\nTraining completed successfully!")
        print(f"Final test accuracy: {test_accuracy * 100:.2f}%")
        
        # Try loading best model
        try:
            best_model_path = os.path.join(model_save_path, 'model_best.h5')
            if os.path.exists(best_model_path):
                print(f"\nLoading best model from {best_model_path}...")
                best_model = tf.keras.models.load_model(best_model_path)
                
                # Prepare test data
                y_test_adjusted = np.zeros((len(y_test), num_classes))
                for i, original_class in enumerate(np.argmax(y_test, axis=1)):
                    if original_class in class_mapping_dict:
                        new_class = class_mapping_dict[original_class]
                        y_test_adjusted[i, new_class] = 1
                
                best_score = best_model.evaluate(X_test, y_test_adjusted, verbose=1)
                print(f"Best model test accuracy: {best_score[1]*100:.2f}%")
            else:
                print("Best model not found.")
        except Exception as e:
            print(f"Error loading best model: {e}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 