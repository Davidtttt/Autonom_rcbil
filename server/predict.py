#!/usr/bin/env python3
"""predict.py: Common prediction functionality for the RC car model.
Used by both autonomous_drive.py and visualize_data.py to ensure consistent predictions."""

import numpy as np
import os
import tensorflow as tf
from image_preprocessing import ImagePreprocessor
import collections
import cv2
from tensorflow.keras.models import load_model

class Predictor:
    def __init__(self, model_path, num_frames=1):
        """Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the model file
            num_frames: Number of frames to use for prediction (default: 1)
        """
        print(f"Loading model from: {model_path}")
        self.model = None
        self.num_frames = 1  # Always use single-frame model
        self.frame_buffer = collections.deque(maxlen=10)  # Fixed: Initialize frame buffer
        
        # Try to load the model with multiple fallback options
        try:
            self.model = load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            try:
                self.model = load_model(model_path, compile=False)
            except Exception:
                # Try alternate model file
                dir_path = os.path.dirname(model_path)
                alt_file = 'model_final.h5' if os.path.basename(model_path) == 'model_best.h5' else 'model_best.h5'
                alt_path = os.path.join(dir_path, alt_file)
                
                if os.path.exists(alt_path):
                    try:
                        self.model = load_model(alt_path, compile=False)
                    except Exception:
                        pass
        
        if self.model:
            print("Model loaded successfully")
            self.model.summary()
        else:
            print("Failed to load any model")
            
        # Create image preprocessor
        self.preprocessor = ImagePreprocessor()
        
        # Command mapping
        self.commands = [b'4', b'3', b'1', b'2']  # Left, Right, Forward, Reverse
    
    def predict_image(self, image):
        """Predict the driving command from a single image."""
        if self.model is None:
            return [[0.25, 0.25, 0.25, 0.25]], 2, b'1'  # Default to forward
        
        processed = self.preprocessor.preprocess(image, output_type='prediction')
        predictions = self.model.predict(processed)
        predicted_class = np.argmax(predictions[0])
        command = self.commands[predicted_class] if predicted_class < len(self.commands) else b'0'
        
        return predictions, predicted_class, command


# For importing
def load_model(model_path=None):
    """Load a trained model if available.
    
    Convenience function for scripts that don't need the full Predictor class.
    """
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'models', 'model_best.h5')
        
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"Loaded model with compile=False")
            return model
        except Exception:
            return None 