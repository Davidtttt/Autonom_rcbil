#!/usr/bin/env python3
"""image_preprocessing.py: Preprocessing functions for RC car images.
Shared between training and autonomous driving scripts."""

import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self, image_height=240, image_width=320):
        """Initialize the image preprocessor with dimensions."""
        self.image_height = image_height
        self.image_width = image_width
        
        # Band-pass filter parameters
        self.lower_threshold = 36  # Filter out very dark pixels (shadows)
        self.upper_threshold = 187  # Filter out very bright pixels (reflections)
        
    def preprocess(self, image, for_training=False):
        """Main preprocessing function that applies all transformations.
        
        Args:
            image: Input grayscale image (full 240x320 camera image)
            for_training: If True, returns flattened array for training
        
        Returns:
            Preprocessed image (rotated, cropped, enhanced contrast)
        """
        # Ensure image is grayscale
        if len(image.shape) > 2 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # 1. Rotate the image 180 degrees (flip it upside down)
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        
        # 2. Take only the upper half after rotation (which was originally the lower half)
        height = rotated.shape[0]
        upper_half = rotated[0:height//2, :]
        
        # 3. Apply high contrast filter for better line detection
        enhanced = self.enhance_contrast(upper_half)
        
        # For display and debugging
        if not for_training:
            # Show the steps for debugging
            self.debug_display(image, rotated, upper_half, enhanced)
            
            # Reshape for model prediction (add channel dimension)
            return enhanced.reshape(1, height//2, self.image_width, 1).astype(np.float32) / 255.0
        
        # For training, flatten and normalize
        return enhanced.flatten().astype(np.float32) / 255.0
    
    def preprocess_roi(self, roi_image, for_training=False):
        """Process ROI images directly as they appear in the training data.
        
        Args:
            roi_image: Input grayscale ROI image (120x320, lower half of camera view)
            for_training: If True, returns flattened array for training
            
        Returns:
            Preprocessed image
        """
        # Ensure image is grayscale
        if len(roi_image.shape) > 2 and roi_image.shape[2] > 1:
            roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            
        # 1. Rotate the ROI image 180 degrees
        rotated_roi = cv2.rotate(roi_image, cv2.ROTATE_180)
        
        # 2. Apply high contrast filter for better line detection
        enhanced = self.enhance_contrast(rotated_roi)
        
        # For model prediction (add channel dimension) or return flattened
        if not for_training:
            return enhanced.reshape(1, roi_image.shape[0], roi_image.shape[1], 1).astype(np.float32) / 255.0
        
        # For training, flatten and normalize
        return enhanced.flatten().astype(np.float32) / 255.0
    
    def enhance_contrast(self, image):
        """Apply edge detection to identify the black line."""
        # Ensure the image is uint8 type for edge detection
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # 1. Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Ensure blurred is uint8 type before Canny
        if blurred.dtype != np.uint8:
            blurred = blurred.astype(np.uint8)
        
        # 2. Apply Canny edge detection
        # Parameters: image, threshold1, threshold2
        edges = cv2.Canny(blurred, 30, 98)
        
        # 3. Dilate the edges to make them more prominent
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 4. Create a mask to identify the line area (dark regions)
        _, line_mask = cv2.threshold(blurred, 74, 255, cv2.THRESH_BINARY_INV)
        
        # 5. Combine edge detection with line area for better line detection
        combined = cv2.bitwise_or(dilated_edges, line_mask)
        
        # 6. Clean up with morphological operations
        # Fill small holes in the line
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 7. Remove small noise
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return combined
    
    def debug_display(self, original, rotated, upper_half, enhanced):
        """Display intermediate preprocessing steps for debugging.
        Only used in development, can be disabled in production."""
        # Uncomment to enable debugging display
        # cv2.imshow('Original', original)
        # cv2.imshow('Rotated', rotated)
        # cv2.imshow('Upper Half', upper_half)
        # cv2.imshow('Edge Detection', enhanced)
        # cv2.waitKey(1)
        pass
    
    def preprocess_for_display(self, image):
        """Preprocess the image for display purposes only."""
        # Ensure image is grayscale
        if len(image.shape) > 2 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Ensure image is uint8 type
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
            
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        height = rotated.shape[0]
        upper_half = rotated[0:height//2, :]
        
        # Apply blur as done in edge detection
        blurred = cv2.GaussianBlur(upper_half, (5, 5), 0)
        
        # Ensure blurred is uint8 type before Canny
        if blurred.dtype != np.uint8:
            blurred = blurred.astype(np.uint8)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 30, 98)
        
        # Get final processed image
        enhanced = self.enhance_contrast(upper_half)
        
        # Create a visual representation of the preprocessing steps
        # Convert grayscale images to BGR for visualization
        original_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        upper_half_color = cv2.cvtColor(upper_half, cv2.COLOR_GRAY2BGR)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Resize all images to the same height for display
        display_height = 120
        original_resized = cv2.resize(original_color, (int(original_color.shape[1] * display_height / original_color.shape[0]), display_height))
        upper_half_resized = cv2.resize(upper_half_color, (int(upper_half_color.shape[1] * display_height / upper_half_color.shape[0]), display_height))
        edges_resized = cv2.resize(edges_color, (int(edges_color.shape[1] * display_height / edges_color.shape[0]), display_height))
        enhanced_resized = cv2.resize(enhanced_color, (int(enhanced_color.shape[1] * display_height / enhanced_color.shape[0]), display_height))
        
        # Create a blank canvas with labels
        canvas_width = original_resized.shape[1] + upper_half_resized.shape[1] + edges_resized.shape[1] + enhanced_resized.shape[1] + 30
        canvas = np.ones((display_height + 30, canvas_width, 3), dtype=np.uint8) * 255
        
        # Add images to canvas
        x_offset = 0
        canvas[15:15+display_height, x_offset:x_offset+original_resized.shape[1]] = original_resized
        cv2.putText(canvas, "Original", (x_offset, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        x_offset += original_resized.shape[1] + 10
        canvas[15:15+display_height, x_offset:x_offset+upper_half_resized.shape[1]] = upper_half_resized
        cv2.putText(canvas, "Upper Half", (x_offset, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        x_offset += upper_half_resized.shape[1] + 10
        canvas[15:15+display_height, x_offset:x_offset+edges_resized.shape[1]] = edges_resized
        cv2.putText(canvas, "Edge Detection", (x_offset, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        x_offset += edges_resized.shape[1] + 10
        canvas[15:15+display_height, x_offset:x_offset+enhanced_resized.shape[1]] = enhanced_resized
        cv2.putText(canvas, "Final Image", (x_offset, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return canvas
    
    def preprocess_roi_for_display(self, roi_image):
        """Preprocess an ROI image for display purposes only."""
        # Ensure image is grayscale
        if len(roi_image.shape) > 2 and roi_image.shape[2] > 1:
            roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        # Ensure image is uint8 type
        if roi_image.dtype != np.uint8:
            roi_image = roi_image.astype(np.uint8)
            
        # Rotate the ROI image 180 degrees
        rotated_roi = cv2.rotate(roi_image, cv2.ROTATE_180)
        
        # Apply blur as in the edge detection process
        blurred = cv2.GaussianBlur(rotated_roi, (5, 5), 0)
        
        # Ensure blurred is uint8 type before Canny
        if blurred.dtype != np.uint8:
            blurred = blurred.astype(np.uint8)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 30, 98)
        
        # Get final edge processed image
        enhanced = self.enhance_contrast(rotated_roi)
        
        # Create a visual representation of the preprocessing steps
        # Convert grayscale images to BGR for visualization
        original_color = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        rotated_color = cv2.cvtColor(rotated_roi, cv2.COLOR_GRAY2BGR)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Resize all images to the same height for display
        display_height = 120
        original_resized = cv2.resize(original_color, (int(original_color.shape[1] * display_height / original_color.shape[0]), display_height))
        rotated_resized = cv2.resize(rotated_color, (int(rotated_color.shape[1] * display_height / rotated_color.shape[0]), display_height))
        edges_resized = cv2.resize(edges_color, (int(edges_color.shape[1] * display_height / edges_color.shape[0]), display_height))
        enhanced_resized = cv2.resize(enhanced_color, (int(enhanced_color.shape[1] * display_height / enhanced_color.shape[0]), display_height))
        
        # Create a blank canvas with labels
        canvas_width = original_resized.shape[1] + rotated_resized.shape[1] + edges_resized.shape[1] + enhanced_resized.shape[1] + 30
        canvas = np.ones((display_height + 30, canvas_width, 3), dtype=np.uint8) * 255
        
        # Add images to canvas
        x_offset = 0
        canvas[15:15+display_height, x_offset:x_offset+original_resized.shape[1]] = original_resized
        cv2.putText(canvas, "Original ROI", (x_offset, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        x_offset += original_resized.shape[1] + 10
        canvas[15:15+display_height, x_offset:x_offset+rotated_resized.shape[1]] = rotated_resized
        cv2.putText(canvas, "Rotated ROI", (x_offset, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        x_offset += rotated_resized.shape[1] + 10
        canvas[15:15+display_height, x_offset:x_offset+edges_resized.shape[1]] = edges_resized
        cv2.putText(canvas, "Edge Detection", (x_offset, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        x_offset += edges_resized.shape[1] + 10
        canvas[15:15+display_height, x_offset:x_offset+enhanced_resized.shape[1]] = enhanced_resized
        cv2.putText(canvas, "Final Image", (x_offset, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return canvas
    
    def load_and_preprocess(self, image_path):
        """Load and preprocess an image from file."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return self.preprocess(image)
    
    def preprocess_batch(self, images):
        """Preprocess a batch of images."""
        return np.array([self.preprocess(img, for_training=True) for img in images])
    
    def preprocess_roi_batch(self, roi_images):
        """Preprocess a batch of ROI images from training data."""
        return np.array([self.preprocess_roi(img, for_training=True) for img in roi_images])
    
    def set_filter_thresholds(self, lower=30, upper=220):
        """Adjust the band-pass filter thresholds."""
        self.lower_threshold = lower  # Filter out very dark pixels (shadows)
        self.upper_threshold = upper  # Filter out very bright pixels (reflections)
        print(f"Band-pass filter thresholds set to: {lower}-{upper}")
    
def reshape_for_model(flattened_images, height=120, width=320):
    """Reshape flattened images for CNN input."""
    return flattened_images.reshape(-1, height, width, 1) 