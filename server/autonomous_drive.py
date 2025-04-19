#!/usr/bin/env python3
"""autonomous_drive.py: Loads a trained neural network model and uses it
to autonomously drive the RC car."""

import cv2
import numpy as np
import socket
import struct
import time
import os
import sys
from image_preprocessing import ImagePreprocessor, reshape_for_model
from predict import Predictor
import collections

def load_model_safe(model_path):
    """Safely load a model with better error handling."""
    from tensorflow.keras.models import load_model
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    try:
        print(f"Attempting to load model from: {model_path}")
        model = load_model(model_path)
        if model is not None:
            model.summary()
            return model
    except Exception as e:
        print(f"Error loading model standard way: {e}")
    
    try:
        print("Trying to load the model with compile=False...")
        model = load_model(model_path, compile=False)
        if model is not None:
            model.summary()
            return model
    except Exception as e:
        print(f"Error loading model with compile=False: {e}")
    
    # Try alternate model
    alt_model_path = os.path.join(os.path.dirname(model_path), 
                            'model_final.h5' if 'best' in model_path else 'model_best.h5')
    if os.path.exists(alt_model_path):
        try:
            print(f"Trying alternate model: {alt_model_path}")
            model = load_model(alt_model_path, compile=False)
            if model is not None:
                model.summary()
                return model
        except Exception as e:
            print(f"Error loading alternate model: {e}")
    
    print("Failed to load any model!")
    return None

class AutonomousDrive:
    def __init__(self, model_path=None):
        """Initialize autonomous driving with the trained model."""
        # Set default model path if not provided
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'models', 'model_best.h5')
        
        # Initialize the predictor with the model
        self.predictor = Predictor(model_path)
        
        # Ensure we have a valid model
        if hasattr(self.predictor, 'model') and self.predictor.model is None:
            print("Predictor failed to load a model, trying safe loader...")
            model = load_model_safe(model_path)
            if model is not None:
                self.predictor.model = model
                print("Successfully loaded model with safe loader!")
            else:
                print("WARNING: Could not load any model. Autonomous driving will not work properly!")
        
        # Initialize image preprocessor for display
        self.preprocessor = ImagePreprocessor()
        
        # Videostream (camera from Raspberry Pi)
        self.sock = socket.socket()
        self.sock.bind(('0.0.0.0', 8000))  # For video
        self.sock.listen(1)
        print("Waiting for Raspberry Pi video connection...")
        self.connection = self.sock.accept()[0].makefile('rb')
        print("Raspberry Pi video connected.")

        # Control socket for sending commands to Raspberry Pi
        self.cmd_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.cmd_server_socket.bind(('0.0.0.0', 8001))  # Listen for commands
        self.cmd_server_socket.listen(1)
        print("Waiting for Raspberry Pi control connection...")
        self.cmd_connection, _ = self.cmd_server_socket.accept()
        print("Raspberry Pi control connected.")

        # Control variables
        self.autonomous_mode = True
        
        # Buffer for storing raw frames for visualization
        self.raw_frame_buffer = collections.deque(maxlen=10)  # Store original frames for display
        
        # Variable to store the previous command
        self.previous_command = None
        
        # Start autonomous driving
        self.drive()

    def drive(self):
        """Main autonomous driving loop."""
        print("Starting autonomous driving...")
        
        try:
            frame_count = 0
            start_time = time.time()
            
            # For displaying the processed image
            cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Processed Images', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Previous Frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Processed Images', 1000, 240)
            cv2.resizeWindow('Previous Frame', 320, 240)
            
            while self.autonomous_mode:
                # Read a frame
                image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
                if not image_len:
                    break
                
                recv_bytes = b''
                recv_bytes += self.connection.read(image_len)
                original_image = cv2.imdecode(np.frombuffer(recv_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                
                # Store the frame in our buffer
                self.raw_frame_buffer.append(original_image.copy())
                
                # Process the image using ImagePreprocessor, which now uses the same
                # processing as train_model.py
                processed_display = self.preprocessor.preprocess(original_image, output_type='display')
                
                # Display the original and processed frames
                cv2.imshow('Original Image', original_image)
                cv2.imshow('Processed Images', processed_display)
                
                # Display previous frame if available (for visualization of temporal data)
                if len(self.raw_frame_buffer) > 1:
                    prev_frame = self.raw_frame_buffer[-2]
                    prev_processed = self.preprocessor.preprocess(prev_frame, output_type='display')
                    cv2.imshow('Previous Frame', prev_processed)
                
                try:
                    # Predict command using predictor module
                    # The predictor.predict_image method handles the temporal data internally
                    # and uses the preprocessor for image preparation
                    _, command_index, command = self.predictor.predict_image(original_image)
                    
                    # Send command to Raspberry Pi only if different from previous command
                    if command != self.previous_command:
                        self.cmd_connection.send(command)
                        self.previous_command = command
                        print(f"Sent new command: {command}")
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    # If there's an error, send stop command
                    self.cmd_connection.send(b'0')
                
                frame_count += 1
                
                # Check for key press to exit autonomous mode
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Exiting autonomous mode...")
                    self.cmd_connection.send(b'0')  # Stop the car
                    self.autonomous_mode = False
                
                # Calculate and display FPS
                if frame_count % 10 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    print(f"FPS: {fps:.2f}")
        
        except Exception as e:
            print(f"Error during autonomous driving: {e}")
        
        finally:
            # Clean up
            self.cmd_connection.send(b'0')  # Stop the car
            cv2.destroyAllWindows()
            self.connection.close()
            self.sock.close()
            self.cmd_connection.close()
            self.cmd_server_socket.close()
            print("Autonomous driving stopped.")

if __name__ == "__main__":
    try:
        AutonomousDrive()
    except KeyboardInterrupt:
        print("User interrupted. Stopping...")
    except Exception as e:
        print(f"Error: {e}") 