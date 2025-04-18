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

class AutonomousDrive:
    def __init__(self, model_path=None):
        """Initialize autonomous driving with the trained model."""
        # Set default model path if not provided
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'models', 'model_best.h5')
        
        # Initialize the predictor with the model
        self.predictor = Predictor(model_path)
        
        # Initialize image preprocessor for display
        self.preprocessor = ImagePreprocessor(image_height=240, image_width=320)
        
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
            cv2.resizeWindow('Processed Images', 1000, 240)
            
            while self.autonomous_mode:
                # Read a frame
                image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
                if not image_len:
                    break
                
                recv_bytes = b''
                recv_bytes += self.connection.read(image_len)
                original_image = cv2.imdecode(np.frombuffer(recv_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                
                # Print original image dimensions for debugging
                print(f"Original image dimensions: {original_image.shape}")
                
                # Extract ROI (lower half)
                if original_image.shape[0] >= 240:
                    roi = original_image[120:240, :]
                else:
                    roi = original_image
                
                # Ensure ROI has correct dimensions for the model
                if roi.shape[0] != 120 or roi.shape[1] != 320:
                    print(f"Resizing ROI from {roi.shape} to (120, 320)")
                    roi = cv2.resize(roi, (320, 120))
                
                # Process the ROI for display (including all preprocessing steps)
                processed_display = self.preprocessor.preprocess_roi_for_display(roi)
                
                # Display the original and processed frames
                cv2.imshow('Original Image', original_image)
                cv2.imshow('Processed Images', processed_display)
                
                # Predict command using predictor module
                _, _, command = self.predictor.predict_image(original_image)
                # If command is left/right, send forward+left/right instead
                if command == b'4':  # Left command
                    command = b'5'  # Forward+Left command
                elif command == b'3':  # Right command
                    command = b'6'  # Forward+Right command
                # Send command to Raspberry Pi
                self.cmd_connection.send(command)
                
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