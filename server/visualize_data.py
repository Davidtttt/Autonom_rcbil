#!/usr/bin/env python3
"""visualize_data.py: Visualize the collected training data and test the preprocessing pipeline."""

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_preprocessing import ImagePreprocessor
import argparse
import tensorflow as tf
from predict import load_model, Predictor

def load_sample_data(data_dir='data_set', max_samples=10):
    """Load a few samples from the training data for visualization."""
    # Get all .npz files in the data_set directory
    data_files = glob.glob(os.path.join(data_dir, '*.npz'))
    
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}. Run collect_data.py first.")
    
    # Load data from the first file
    data = np.load(data_files[0])
    images = data['images']
    labels = data['labels']
    
    # Get class names
    class_names = ['Left', 'Right', 'Forward', 'Reverse']
    
    # Sample a few examples of each class
    samples = []
    for class_idx in range(4):
        # Find examples of this class
        class_indices = np.where(labels[:, class_idx] == 1)[0]
        
        # Sample a few (or all if fewer than max_samples)
        sample_count = min(max_samples, len(class_indices))
        if sample_count > 0:
            for i in range(sample_count):
                idx = class_indices[i]
                samples.append({
                    'image': images[idx].reshape(120, 320),
                    'label': class_names[class_idx],
                    'class_idx': class_idx
                })
    
    return samples

def predict_images(model, samples, preprocessor):
    """Add predictions to samples using the loaded model."""
    if model is None:
        return samples
    
    # Create a predictor instance if we're given a model directly
    if not isinstance(model, Predictor):
        predictor = Predictor(model_path=None)
        predictor.model = model  # Use the provided model
    else:
        predictor = model
    
    class_names = ['Left', 'Right', 'Forward', 'Reverse']
    
    for sample in samples:
        # Get the ROI image
        roi_image = sample['image']
        
        # Ensure roi_image is uint8 type
        if roi_image.dtype != np.uint8:
            roi_image = roi_image.astype(np.uint8)
            
        # Preprocess image for model prediction
        processed = preprocessor.preprocess_roi(roi_image, for_training=False)
        
        # Make prediction using predictor
        pred_class_idx, confidence, _ = predictor.predict_without_preprocessing(processed, smooth=False, verbose=False)
        
        # Add prediction to sample
        sample['prediction'] = class_names[pred_class_idx]
        sample['confidence'] = confidence
        sample['is_correct'] = (pred_class_idx == sample['class_idx'])
    
    return samples

def visualize_preprocessing(samples):
    """Visualize the preprocessing steps for a few samples."""
    # Initialize image preprocessor
    preprocessor = ImagePreprocessor(image_height=240, image_width=320)
    
    # Load model if available and create predictor
    model = load_model()
    predictor = None
    if model is not None:
        predictor = Predictor(model_path=None)
        predictor.model = model  # Use the loaded model
    
    # Get predictions if model is available
    if predictor is not None:
        samples = predict_images(predictor, samples, preprocessor)
    
    # Create figure for visualization
    num_samples = min(4, len(samples))
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i in range(num_samples):
        sample = samples[i]
        
        # Get the ROI image directly from training data
        roi_image = sample['image']
        
        # Ensure roi_image is uint8 type
        if roi_image.dtype != np.uint8:
            roi_image = roi_image.astype(np.uint8)
        
        # Process the ROI image directly
        # 1. Rotate the ROI image 180 degrees
        rotated_roi = cv2.rotate(roi_image, cv2.ROTATE_180)
        
        # 2. Apply Gaussian blur
        blurred = cv2.GaussianBlur(rotated_roi, (5, 5), 0)
        
        # Ensure blurred is uint8 type
        if blurred.dtype != np.uint8:
            blurred = blurred.astype(np.uint8)
        
        # 3. Apply Canny edge detection
        edges = cv2.Canny(blurred, 30, 150)
        
        # 4. Apply full edge detection pipeline
        enhanced = preprocessor.enhance_contrast(rotated_roi)
        
        # Create title with ground truth and prediction
        if predictor is not None and 'prediction' in sample:
            title = f"True: {sample['label']}, Predicted: {sample['prediction']} ({sample['confidence']:.2f})"
            # Add color coding for correct/incorrect predictions
            title_color = 'green' if sample['is_correct'] else 'red'
        else:
            title = f"Class: {sample['label']}"
            title_color = 'black'
        
        # Plot the images
        plt.subplot(num_samples, 4, i*4 + 1)
        plt.imshow(roi_image, cmap='gray')
        if i == 0:
            plt.title("Original ROI")
        else:
            plt.title("")
        plt.text(10, 20, title, color=title_color, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i*4 + 2)
        plt.imshow(rotated_roi, cmap='gray')
        if i == 0:
            plt.title('Rotated ROI (180°)')
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i*4 + 3)
        plt.imshow(edges, cmap='gray')
        if i == 0:
            plt.title('Canny Edge Detection')
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i*4 + 4)
        plt.imshow(enhanced, cmap='gray')
        if i == 0:
            plt.title('Final Edge Image')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_visualization.png')
    plt.show()
    
    # Also save a single detailed visualization
    if len(samples) > 0:
        sample = samples[0]
        roi_image = sample['image']
        
        # Ensure roi_image is uint8 type before passing to preprocess_roi_for_display
        if roi_image.dtype != np.uint8:
            roi_image = roi_image.astype(np.uint8)
        
        # Create a detailed visualization with the new preprocess_roi_for_display method
        display_image = preprocessor.preprocess_roi_for_display(roi_image)
        
        # Add title with class information and prediction if available
        if predictor is not None and 'prediction' in sample:
            title = f"ROI Processing Pipeline - True: {sample['label']}, Predicted: {sample['prediction']} ({sample['confidence']:.2f})"
            # Color code based on prediction correctness
            color = (0, 255, 0) if sample['is_correct'] else (0, 0, 255)  # Green if correct, red if wrong
        else:
            title = f"ROI Processing Pipeline - Class: {sample['label']}"
            color = (0, 0, 0)  # Black
            
        cv2.putText(display_image, title, (10, display_image.shape[0] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Save and display the detailed visualization
        cv2.imwrite('detailed_preprocessing.png', display_image)
        
        # Show with matplotlib for better display in notebooks
        plt.figure(figsize=(15, 5))
        plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def visualize_class_distribution(data_dir='data_set'):
    """Visualize the distribution of classes in the training data."""
    # Get all .npz files in the data_set directory
    data_files = glob.glob(os.path.join(data_dir, '*.npz'))
    
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}. Run collect_data.py first.")
    
    # Count examples per class
    class_counts = [0, 0, 0, 0]
    total_samples = 0
    
    for file in data_files:
        data = np.load(file)
        labels = data['labels']
        
        for i in range(4):
            class_counts[i] += np.sum(labels[:, i])
        
        total_samples += labels.shape[0]
    
    # Visualize distribution
    class_names = ['Left', 'Right', 'Forward', 'Reverse']
    
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_counts)
    plt.title('Class Distribution in Training Data')
    plt.xlabel('Class')
    plt.ylabel('Number of Examples')
    
    # Add count and percentage labels
    for i, count in enumerate(class_counts):
        percentage = (count / total_samples) * 100
        plt.text(i, count + 5, f"{int(count)}\n({percentage:.1f}%)", 
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()
    
    print(f"Total samples: {total_samples}")
    print(f"Image dimensions: {data['images'][0].reshape(120, 320).shape}")
    for i, name in enumerate(class_names):
        print(f"{name}: {class_counts[i]} ({(class_counts[i]/total_samples)*100:.1f}%)")

def interactive_parameter_tuning(image_path=None):
    """Interactive tool to tune the preprocessing parameters."""
    # Initialize preprocessing parameters
    lower_threshold = 30  # Canny lower threshold
    upper_threshold = 150  # Canny upper threshold
    thresh_value = 100  # Threshold for binary mask
    
    preprocessor = ImagePreprocessor(image_height=240, image_width=320)
    
    # Load model if available and create predictor
    model = load_model()
    predictor = None
    if model is not None:
        predictor = Predictor(model_path=None)
        predictor.model = model  # Use the loaded model
    
    class_names = ['Left', 'Right', 'Forward', 'Reverse']
    
    # Try to use training data for tuning if available
    use_training_data = False
    training_samples = None
    
    try:
        if os.path.exists('data_set'):
            data_files = glob.glob(os.path.join('data_set', '*.npz'))
            if data_files:
                print("Found training data, using it for parameter tuning.")
                training_samples = load_sample_data(max_samples=5)
                if training_samples:
                    use_training_data = True
                    print(f"Loaded {len(training_samples)} training samples.")
                    
                    # Add predictions if model is available
                    if predictor is not None:
                        training_samples = predict_images(predictor, training_samples, preprocessor)
    except Exception as e:
        print(f"Could not load training data: {e}")
    
    # If no image path provided and no training data, try to find a collected image or use webcam
    if not use_training_data and image_path is None:
        # Try to find collected images
        if os.path.exists('collected_images'):
            image_files = glob.glob(os.path.join('collected_images', '*.jpg'))
            if image_files:
                image_path = image_files[0]
                print(f"Using collected image: {image_path}")
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None and image.shape[0] > 120:
                    # Extract ROI from camera image if needed
                    image = image[120:240, :]  # Use lower half as ROI
            else:
                # No images found, try to use webcam
                print("No collected images found. Trying webcam...")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Error: Could not open webcam. Please provide an image path.")
                    return
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read from webcam.")
                    return
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Extract ROI from camera image
                if image.shape[0] > 120:
                    image = image[-120:, :]  # Use lower portion as ROI
                cap.release()
        else:
            print("No collected_images directory found. Please provide an image path.")
            return
    elif not use_training_data:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None and image.shape[0] > 120:
            # Extract ROI from camera image if needed
            image = image[120:240, :]  # Use lower half as ROI
        
    # For non-training data, check if image was loaded successfully
    if not use_training_data and image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Create windows for controls and results
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 600, 200)
    cv2.namedWindow('Processed Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Processed Image', 900, 300)
    
    # Sample index for training data
    sample_idx = 0
    
    # Create a function to update the display based on current parameters
    def update_display():
        # Apply preprocessing pipeline
        preprocessor.set_filter_thresholds(lower_threshold, upper_threshold)
        
        # Make a copy of the enhance_contrast method with custom threshold
        def custom_enhance_contrast(img):
            # Ensure input is uint8 type
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
                
            # 1. Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Ensure blurred is uint8 type
            if blurred.dtype != np.uint8:
                blurred = blurred.astype(np.uint8)
            
            # 2. Apply Canny edge detection with adjustable parameters
            edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
            
            # 3. Dilate the edges to make them more prominent
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            # 4. Create a mask to identify the line area (dark regions)
            _, line_mask = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY_INV)
            
            # 5. Combine edge detection with line area for better line detection
            combined = cv2.bitwise_or(dilated_edges, line_mask)
            
            # 6. Clean up with morphological operations
            # Fill small holes in the line
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # 7. Remove small noise
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
            
            return combined
        
        if use_training_data:
            # Use training data sample (already ROI formatted)
            nonlocal sample_idx
            sample = training_samples[sample_idx]
            roi_image = sample['image'].copy()
            class_label = sample['label']
            
            # If we have a model and predictions, get the prediction
            prediction_text = ""
            prediction_color = (0, 0, 0)  # Black by default
            if predictor is not None and 'prediction' in sample:
                prediction_text = f", Predicted: {sample['prediction']} ({sample['confidence']:.2f})"
                prediction_color = (0, 255, 0) if sample['is_correct'] else (0, 0, 255)  # Green if correct, red if wrong
                
                # Also make a new prediction with current parameters to see if it improves
                processed_for_model = preprocessor.preprocess_roi(roi_image, for_training=False)
                current_pred_idx, current_confidence, _ = predictor.predict_without_preprocessing(processed_for_model, smooth=False, verbose=False)
                current_pred_label = class_names[current_pred_idx]
                is_current_correct = (current_pred_idx == sample['class_idx'])
                
                # Add current prediction with these parameters
                prediction_text += f"\nWith current params: {current_pred_label} ({current_confidence:.2f})"
                if is_current_correct != sample['is_correct']:
                    if is_current_correct:
                        prediction_text += " ✓"  # Show improvement
                    else:
                        prediction_text += " ✗"  # Show worsening
        else:
            # Use the loaded image (already converted to ROI format)
            roi_image = image.copy()
            class_label = "Unknown"
            prediction_text = ""
            prediction_color = (0, 0, 0)
            
            # If we have a model, make a prediction on this image
            if predictor is not None:
                # Make a prediction with current parameters
                processed_for_model = preprocessor.preprocess_roi(roi_image, for_training=False)
                pred_class_idx, confidence, _ = predictor.predict_without_preprocessing(processed_for_model, smooth=False, verbose=False)
                pred_label = class_names[pred_class_idx]
                
                prediction_text = f", Predicted: {pred_label} ({confidence:.2f})"
        
        # Ensure roi_image is uint8 type
        if roi_image.dtype != np.uint8:
            roi_image = roi_image.astype(np.uint8)
            
        # Apply direct ROI processing:
        # 1. Rotate the ROI 180 degrees
        rotated_roi = cv2.rotate(roi_image, cv2.ROTATE_180)
        
        # 2. Apply Gaussian blur as in edge detection
        blurred = cv2.GaussianBlur(rotated_roi, (5, 5), 0)
        
        # Ensure blurred is uint8 type before Canny
        if blurred.dtype != np.uint8:
            blurred = blurred.astype(np.uint8)
        
        # 3. Apply Canny edge detection
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
        
        # 4. Apply full edge detection
        enhanced = custom_enhance_contrast(rotated_roi)
        
        # Convert grayscale images to BGR for visualization
        original_color = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        rotated_color = cv2.cvtColor(rotated_roi, cv2.COLOR_GRAY2BGR)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Resize all images for the display panel
        display_height = 120
        original_resized = cv2.resize(original_color, (int(original_color.shape[1] * display_height / original_color.shape[0]), display_height))
        rotated_resized = cv2.resize(rotated_color, (int(rotated_color.shape[1] * display_height / rotated_color.shape[0]), display_height))
        edges_resized = cv2.resize(edges_color, (int(edges_color.shape[1] * display_height / edges_color.shape[0]), display_height))
        enhanced_resized = cv2.resize(enhanced_color, (int(enhanced_color.shape[1] * display_height / enhanced_color.shape[0]), display_height))
        
        # Create a blank canvas for the visualization panel
        panel_width = original_resized.shape[1] + rotated_resized.shape[1] + edges_resized.shape[1] + enhanced_resized.shape[1] + 40
        panel_height = display_height + 80 if prediction_text else display_height + 60
        panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255
        
        # Add images to panel
        x_offset = 10
        panel[40:40+display_height, x_offset:x_offset+original_resized.shape[1]] = original_resized
        cv2.putText(panel, "Original ROI", (x_offset, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        x_offset += original_resized.shape[1] + 10
        panel[40:40+display_height, x_offset:x_offset+rotated_resized.shape[1]] = rotated_resized
        cv2.putText(panel, "Rotated ROI", (x_offset, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        x_offset += rotated_resized.shape[1] + 10
        panel[40:40+display_height, x_offset:x_offset+edges_resized.shape[1]] = edges_resized
        cv2.putText(panel, "Edge Detection", (x_offset, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        x_offset += edges_resized.shape[1] + 10
        panel[40:40+display_height, x_offset:x_offset+enhanced_resized.shape[1]] = enhanced_resized
        cv2.putText(panel, "Final Image", (x_offset, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add title and class information
        title = f"Direct ROI Processing - True: {class_label}{prediction_text}"
        cv2.putText(panel, title, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, prediction_color, 1)
        
        # If we have additional prediction info, add it on another line
        if "\n" in prediction_text:
            lines = prediction_text.split("\n")
            second_line = f"True: {class_label}{lines[1]}"
            cv2.putText(panel, second_line, (10, panel.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, prediction_color, 1)
        
        # Display parameters on the panel
        param_info = f"Parameters: Lower={lower_threshold} | Upper={upper_threshold} | Threshold={thresh_value}"
        cv2.putText(panel, param_info, (10, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Display the visualization panel
        cv2.imshow('Processed Image', panel)
        
        # Also display just the enhanced output for closer inspection
        cv2.namedWindow('Enhanced Output', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Enhanced Output', 400, 300)
        cv2.imshow('Enhanced Output', enhanced)
    
    # Create trackbars for parameters
    def on_lower_threshold_change(val):
        nonlocal lower_threshold
        lower_threshold = val
        update_display()
    
    def on_upper_threshold_change(val):
        nonlocal upper_threshold
        upper_threshold = val
        update_display()
    
    def on_thresh_value_change(val):
        nonlocal thresh_value
        thresh_value = val
        update_display()
    
    cv2.createTrackbar('Canny Lower Threshold', 'Controls', lower_threshold, 255, on_lower_threshold_change)
    cv2.createTrackbar('Canny Upper Threshold', 'Controls', upper_threshold, 255, on_upper_threshold_change)
    cv2.createTrackbar('Binary Threshold', 'Controls', thresh_value, 255, on_thresh_value_change)
    
    # Initial display update
    update_display()
    
    print("\nInteractive Parameter Tuning")
    print("===========================")
    print("Use the sliders to adjust:")
    print("  - Canny Lower Threshold: controls edge detection sensitivity (lower = more edges)")
    print("  - Canny Upper Threshold: controls edge detection precision (higher = less noise)")
    print("  - Binary Threshold: controls the threshold for detecting dark areas (the line)")
    if use_training_data:
        print("\nNavigation Keys:")
        print("  - Next sample: 'n', 'k', 'd', SPACE, RIGHT ARROW, or DOWN ARROW")
        print("  - Previous sample: 'p', 'a', LEFT ARROW, or UP ARROW")
    print("Press 's' to save the current settings to image_preprocessing.py")
    print("Press 'q' to quit without saving")
    
    # Main loop
    while True:
        key = cv2.waitKey(100) & 0xFF
        
        # Debug output for key detection
        if key != 255:  # Skip if no key is pressed (255 is usually the value when no key is pressed)
            print(f"Key pressed: {key} (ASCII: {chr(key) if 32 <= key <= 126 else 'non-printable'})")
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the current parameters
            print(f"\nSaving parameters: Canny Lower={lower_threshold}, Canny Upper={upper_threshold}, Binary Threshold={thresh_value}")
            
            # Update the enhance_contrast method in image_preprocessing.py
            with open('server/image_preprocessing.py', 'r') as file:
                lines = file.readlines()
            
            # Update parameter values in the file
            for i, line in enumerate(lines):
                if "edges = cv2.Canny(blurred, " in line or "edges = cv2.Canny(contrast_enhanced, " in line:
                    lines[i] = f"        edges = cv2.Canny(blurred, {lower_threshold}, {upper_threshold})\n"
                elif "_, line_mask = cv2.threshold(blurred, " in line or "_, line_mask = cv2.threshold(contrast_enhanced, " in line:
                    lines[i] = f"        _, line_mask = cv2.threshold(blurred, {thresh_value}, 255, cv2.THRESH_BINARY_INV)\n"
            
            # Write the updated file
            with open('server/image_preprocessing.py', 'w') as file:
                file.writelines(lines)
            
            print("Parameters saved to image_preprocessing.py")
            break
        
        # Check for next sample keys
        elif use_training_data and (
            key == ord('n') or        # n
            key == ord('k') or        # k
            key == 110 or             # n (ASCII)
            key == 78 or              # N (ASCII)
            key == 83 or              # RIGHT ARROW on some systems
            key == 84 or              # DOWN ARROW on some systems
            key == 100 or             # d (ASCII)
            key == 68 or              # D (ASCII)
            key == 32 or              # SPACE
            key == 2555904 or         # RIGHT ARROW (WinAPI)
            key == 2621440 or         # DOWN ARROW (WinAPI)
            key == 65363 or           # RIGHT ARROW (Linux GTK)
            key == 65364              # DOWN ARROW (Linux GTK)
        ):
            # Cycle to next training sample
            sample_idx = (sample_idx + 1) % len(training_samples)
            print(f"Showing sample {sample_idx+1}/{len(training_samples)}")
            update_display()
        
        # Check for previous sample keys
        elif use_training_data and (
            key == ord('p') or        # p
            key == 112 or             # p (ASCII)
            key == 80 or              # P (ASCII)
            key == 81 or              # LEFT ARROW on some systems
            key == 82 or              # UP ARROW on some systems
            key == 97 or              # a (ASCII)
            key == 65 or              # A (ASCII)
            key == 2424832 or         # LEFT ARROW (WinAPI)
            key == 2490368 or         # UP ARROW (WinAPI)
            key == 65361 or           # LEFT ARROW (Linux GTK)
            key == 65362              # UP ARROW (Linux GTK)
        ):
            # Cycle to previous training sample
            sample_idx = (sample_idx - 1) % len(training_samples)
            print(f"Showing sample {sample_idx+1}/{len(training_samples)}")
            update_display()
    
    cv2.destroyAllWindows()

def main():
    """Main function to run visualizations."""
    parser = argparse.ArgumentParser(description='Visualize and tune preprocessing for RC car images')
    parser.add_argument('--tune', action='store_true', help='Run interactive parameter tuning')
    parser.add_argument('--image', type=str, help='Path to image for parameter tuning')
    parser.add_argument('--class-stats', action='store_true', help='Show class distribution stats only')
    parser.add_argument('--model', type=str, default='models/model_final.h5', help='Path to model file')
    args = parser.parse_args()
    
    # First check if we have available data
    has_data = False
    try:
        if os.path.exists('data_set'):
            data_files = glob.glob(os.path.join('data_set', '*.npz'))
            has_data = len(data_files) > 0
            if has_data:
                print(f"Found {len(data_files)} data files to analyze.")
    except Exception as e:
        print(f"Error checking for data: {e}")
    
    if args.class_stats and has_data:
        # Just show the class distribution
        try:
            visualize_class_distribution()
        except Exception as e:
            print(f"Error: {e}")
        return
    
    if args.tune:
        # Run parameter tuning with available data
        interactive_parameter_tuning(args.image)
    elif has_data:
        try:
            # Load sample data
            samples = load_sample_data()
            
            # Visualize preprocessing
            visualize_preprocessing(samples)
            
            # Visualize class distribution
            visualize_class_distribution()
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No data available. Please collect training data first or use --image to specify an image.")
        print("Waiting for Raspberry Pi video connection...")

if __name__ == "__main__":
    main() 