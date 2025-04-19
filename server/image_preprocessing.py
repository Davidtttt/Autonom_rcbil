#!/usr/bin/env python3
"""image_preprocessing.py: Preprocessing functions for RC car images.
Shared between training and autonomous driving scripts."""

import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self):
        """Initialize the image preprocessor with dimensions."""
        
    def preprocess(self, image, output_type='default'):
        """Unified preprocessing method for all use cases.
        
        Args:
            image: Input grayscale image (full size 240x320)
            output_type: 
                'default': Returns processed image in original dimensions
                'training': Returns flattened array for training
                'display': Returns visualization for UI display
                'prediction': Returns image formatted for model prediction
                
        Returns:
            Processed image in the requested format
        """
        # Ensure image is grayscale
        if len(image.shape) > 2 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        
        # Validate image dimensions
        if image.shape[0] < 240 or image.shape[1] < 320:
            raise ValueError(f"Image dimensions {image.shape} are too small. Expected minimum dimensions of 240x320.")
            
        # Ensure image is uint8 type
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
        # Step 1: Rotate the image 180 degrees so line is in front of car
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        
        # Step 2: Take only bottom half - where the line should be
        roi = rotated[120:240, :]
        
        # Process image using the same approach as train_model.py
        processed = self._enhance_line_detection(roi)
        
        # Handle different output types
        if output_type == 'training':
            # For training: flatten and normalize
            return processed.flatten().astype(np.float32) / 255.0
            
        elif output_type == 'prediction':
            # For prediction: reshape with channel dimension
            return processed.reshape(1, processed.shape[0], processed.shape[1], 1).astype(np.float32) / 255.0
            
        elif output_type == 'display':
            # Create visualization of processing steps
            return self._create_display_image(image, roi, processed)
            
        else:
            # Default: just return the processed image
            return processed
    
    def _enhance_line_detection(self, roi):
        """Enhanced preprocessing focusing on line detection, matching final_model.py."""
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
            # Create a synthetic line as fallback
            # Just add a vertical line in the middle
            center_x = line_mask.shape[1] // 2
            line_mask[:, center_x-5:center_x+5] = 255
        
        return line_mask
    
    def _create_display_image(self, original, roi, processed):
        """Create visualization showing processing steps."""
        # Create a visualization with original, ROI, and processed image
        # to match the display from train_model.py
        display_width = 1000
        display_height = 240
        
        # Create display canvas (white background)
        display = np.ones((display_height, display_width, 3), dtype=np.uint8) * 255
        
        # Convert grayscale images to color for display
        original_color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        processed_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        # Calculate dimensions for display components
        img_width = min(310, display_width // 3 - 20)
        img_height = 230  # Fixed height to avoid broadcasting issues
        
        # Position the images in the display - use resize with exact dimensions
        # Original
        resized_original = cv2.resize(original_color, (img_width, img_height))
        display[10:10+img_height, 10:10+img_width] = resized_original
        cv2.putText(display, "Original", (10, img_height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # ROI - half height for bottom half of image
        roi_height = img_height // 2
        resized_roi = cv2.resize(roi_color, (img_width, roi_height))
        display[10:10+roi_height, 20+img_width:20+2*img_width] = resized_roi
        cv2.putText(display, "ROI", (20+img_width, roi_height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Processed
        resized_processed = cv2.resize(processed_color, (img_width, roi_height))
        display[10:10+roi_height, 30+2*img_width:30+3*img_width] = resized_processed
        cv2.putText(display, "Processed", (30+2*img_width, roi_height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return display
    
    def preprocess_batch(self, images, output_type='training'):
        """Preprocess a batch of images.
        
        Args:
            images: List or array of input images
            output_type: The output format needed ('training', 'prediction', 'display', 'default')
                
        Returns:
            Array of processed images in the requested format
        """
        return np.array([self.preprocess(img, output_type=output_type) for img in images])
    
def reshape_for_model(flattened_images, height=120, width=320):
    """Reshape flattened images for CNN input."""
    return flattened_images.reshape(-1, height, width, 1) 