#!/usr/bin/env python3
"""visualize_training_data.py: Simple tool to visualize training data the way the neural network sees it,
together with the prediction."""

import cv2
import numpy as np
import os
import glob
import sys
from image_preprocessing import ImagePreprocessor
from predict import Predictor, load_model

def load_data_from_npz(data_dir='data_set'):
    """Load images and labels from NPZ files."""
    images_list = []
    labels_list = []
    data_files = glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_dir, '*.npz'))
    
    if not data_files:
        print(f"No data files found in {data_dir}.")
        return images_list, labels_list
    
    # Load all npz files
    for file in data_files:
        try:
            print(f"Loading data from {file}...")
            data = np.load(file)
            npz_images = data['images']
            npz_labels = data['labels']
            
            # Reshape images from flattened arrays to 2D images
            full_height = 240
            full_width = 320
            
            for i in range(npz_images.shape[0]):
                # Convert normalized float data back to uint8 if needed
                img = npz_images[i].reshape(full_height, full_width)
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                
                label_idx = np.argmax(npz_labels[i])
                images_list.append(img)
                labels_list.append(label_idx)
                
        except Exception as e:
            print(f"Error loading data from {file}: {e}")
    
    print(f"Loaded {len(images_list)} images with labels from NPZ files")
    return images_list, labels_list

def main():
    # Hardcoded paths
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model_best.h5')
    data_dir = 'data_set'  # Directory containing NPZ files
    
    # Load the model
    model = load_model(model_path)
    if model is None:
        print("Failed to load model!")
        sys.exit(1)
    
    # Create the predictor
    predictor = Predictor(model_path)
    
    # Create the image preprocessor
    preprocessor = ImagePreprocessor()
    
    # Command names for display
    command_names = ['LEFT', 'RIGHT', 'FORWARD', 'REVERSE']
    
    # Load images and labels from NPZ files
    images, labels = load_data_from_npz(data_dir)
    if not images:
        print(f"No images found in {data_dir} NPZ files")
        sys.exit(1)
    
    print(f"Found {len(images)} images")
    
    # Set up display windows
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Network Input', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Processing Steps', cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow('Original Image', 640, 480)
    cv2.resizeWindow('Network Input', 320, 120)
    cv2.resizeWindow('Processing Steps', 1000, 240)
    
    # Current image index
    current_idx = 0
    
    while True:
        # Get the current image and label
        original = images[current_idx]
        label_idx = labels[current_idx]
        
        # Create a copy for display
        display_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        
        # Get processing visualization (shows all steps)
        processing_viz = preprocessor.preprocess(original, output_type='display')
        
        # Get the raw processed image (line detection output)
        processed_img = preprocessor.preprocess(original, output_type='default')
        
        # Convert processed image to 3-channel for display
        processed_display = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        
        # Get prediction
        try:
            prediction, command_idx, command = predictor.predict_image(original)
            
            # Display prediction on original image
            cmd_text = command_names[command_idx]
            confidence = prediction[0][command_idx] * 100
            
            # Add text to display image
            cv2.putText(display_img, f"Prediction: {cmd_text}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_img, f"Confidence: {confidence:.1f}%", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show image index
            cv2.putText(display_img, f"Image: {current_idx+1}/{len(images)}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display true label
            true_label_text = command_names[label_idx]
            cv2.putText(display_img, f"True Label: {true_label_text}", (10, 150), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Highlight if prediction matches label
            if label_idx == command_idx:
                cv2.putText(display_img, "MATCH", (10, 190), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_img, "MISMATCH", (10, 190), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            cv2.putText(display_img, "ERROR: Prediction failed", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display images
        cv2.imshow('Original Image', display_img)
        cv2.imshow('Network Input', processed_display)
        cv2.imshow('Processing Steps', processing_viz)
        
        # Print navigation instructions once
        if current_idx == 0:
            print("\nNavigation Controls:")
            print("  Space/n: Next image")
            print("  p/b: Previous image") 
            print("  q: Quit")
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        # Navigation controls
        if key == ord('q'):  # Quit
            break
        elif key == ord('n') or key == ord(' '):  # Next image
            current_idx = (current_idx + 1) % len(images)
        elif key == ord('p') or key == ord('b'):  # Previous image
            current_idx = (current_idx - 1) % len(images)
    
    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 