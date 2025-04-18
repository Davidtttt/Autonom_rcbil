#!/usr/bin/env python3
"""predict.py: Common prediction functionality for the RC car model.
Used by both autonomous_drive.py and visualize_data.py to ensure consistent predictions."""

import numpy as np
import os
import tensorflow as tf
from image_preprocessing import ImagePreprocessor

class Predictor:
    def __init__(self, model_path='models/model_best.h5'):
        """Initialize the predictor with the trained model."""
        # Load the trained model
        if model_path is not None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}. Run train_model.py first.")
            
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded: {model_path}")
        else:
            # Model will be set externally
            self.model = None
            print("Predictor initialized without a model. Use predictor.model = model to set a model.")
        
        # Initialize image preprocessor
        self.preprocessor = ImagePreprocessor(image_height=240, image_width=320)
        
        # Class names for prediction output
        self.class_names = ['LEFT', 'RIGHT', 'FORWARD', 'REVERSE']
        
        # Mapping from prediction index to command
        self.commands = {
            0: b'4',  # Left
            1: b'3',  # Right
            2: b'1',  # Forward
            3: b'2',  # Reverse
        }
        
        # Initialize a buffer for smoothing commands
        self.command_buffer = []
        self.buffer_size = 3  # Number of frames to average
    
    def predict_image(self, image, smooth=True, verbose=True):
        """Predict the driving command based on the image.
        
        Args:
            image: Input grayscale image (full image or ROI)
            smooth: Whether to use command smoothing to prevent rapid changes
            verbose: Whether to print prediction details
            
        Returns:
            Tuple of (command_index, confidence, command_byte)
        """
        # Extract ROI from the camera image (lower half) if needed
        if image.shape[0] >= 240:
            roi = image[120:240, :]
        else:
            # If image is already ROI-sized, use as is
            roi = image
        
        # Ensure the ROI dimensions match what the model expects (120x320)
        if roi.shape[0] != 120 or roi.shape[1] != 320:
            if verbose:
                print(f"Resizing ROI from {roi.shape} to (120, 320)")
            roi = cv2.resize(roi, (320, 120))
        
        # Preprocess the ROI to match the expected input format
        processed = self.preprocessor.preprocess_roi(roi, for_training=False)
        
        # Model prediction
        prediction = self.model.predict(processed, verbose=0)[0]
        
        # Get the index of the highest probability
        command_index = np.argmax(prediction)
        confidence = prediction[command_index]
        
        if smooth:
            # Add to command buffer for smoothing
            self.command_buffer.append((command_index, confidence))
            if len(self.command_buffer) > self.buffer_size:
                self.command_buffer.pop(0)
            
            # Implement simple command smoothing (prevent rapid switching)
            if len(self.command_buffer) >= self.buffer_size:
                # Count occurrences of each command in the buffer
                command_counts = {}
                for cmd, conf in self.command_buffer:
                    if cmd not in command_counts:
                        command_counts[cmd] = 0
                    command_counts[cmd] += 1
                
                # Get the most frequent command
                most_frequent_cmd = max(command_counts, key=command_counts.get)
                
                # Only change command if it's consistently predicted
                if command_counts[most_frequent_cmd] >= self.buffer_size // 2 + 1:
                    command_index = most_frequent_cmd
        
        # Map to command byte
        command = self.commands[command_index]
        
        # Print the prediction with confidence if verbose
        if verbose:
            print(f"Predicted: {self.class_names[command_index]} (Confidence: {confidence:.4f})")
        
        return command_index, confidence, command
    
    def predict_without_preprocessing(self, processed_image, smooth=True, verbose=True):
        """Make prediction on an already preprocessed image.
        
        Used when the preprocessing has already been done (e.g., in training data).
        
        Args:
            processed_image: Already preprocessed image ready for model input
            smooth: Whether to use command smoothing
            verbose: Whether to print prediction details
            
        Returns:
            Tuple of (command_index, confidence, command_byte)
        """
        # Model prediction
        prediction = self.model.predict(processed_image, verbose=0)[0]
        
        # Get the index of the highest probability
        command_index = np.argmax(prediction)
        confidence = prediction[command_index]
        
        if smooth:
            # Add to command buffer for smoothing
            self.command_buffer.append((command_index, confidence))
            if len(self.command_buffer) > self.buffer_size:
                self.command_buffer.pop(0)
            
            # Implement simple command smoothing (prevent rapid switching)
            if len(self.command_buffer) >= self.buffer_size:
                # Count occurrences of each command in the buffer
                command_counts = {}
                for cmd, conf in self.command_buffer:
                    if cmd not in command_counts:
                        command_counts[cmd] = 0
                    command_counts[cmd] += 1
                
                # Get the most frequent command
                most_frequent_cmd = max(command_counts, key=command_counts.get)
                
                # Only change command if it's consistently predicted
                if command_counts[most_frequent_cmd] >= self.buffer_size // 2 + 1:
                    command_index = most_frequent_cmd
        
        # Map to command byte
        command = self.commands[command_index]
        
        # Print the prediction with confidence if verbose
        if verbose:
            print(f"Predicted: {self.class_names[command_index]} (Confidence: {confidence:.4f})")
        
        return command_index, confidence, command
    
    def reset_buffer(self):
        """Reset the command buffer for a fresh start."""
        self.command_buffer = []


# For importing
import cv2  # This was missing in the function calls above

def load_model(model_path='models/model_best.h5'):
    """Load a trained model if available.
    
    Convenience function for scripts that don't need the full Predictor class.
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Will not show predictions.")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None 