# Autonomous RC Car with Machine Learning

This project implements an autonomous RC car using machine learning. The system uses a Raspberry Pi with camera for video streaming and motor control, and a server computer for processing video data, training the model, and controlling the car.

## System Overview

The system consists of four main components:

1. **Data Collection** - Collect training data by manually driving the car.
2. **Data Preprocessing** - Process images with rotation, cropping, and contrast enhancement.
3. **Model Training** - Train a neural network model on the preprocessed data.
4. **Autonomous Driving** - Use the trained model to drive the car autonomously.

## Prerequisites

### Server Computer
- Python 3.6+
- OpenCV
- NumPy
- TensorFlow 2.x
- Matplotlib
- scikit-learn
- pygame

Install the required packages:
```bash
pip install opencv-python numpy tensorflow matplotlib scikit-learn pygame
```

### Raspberry Pi
- Raspberry Pi with camera module
- Python 3
- picamera2
- RPi.GPIO
- Motor driver circuit (connected to GPIO pins)

Install the required packages:
```bash
pip install picamera2 RPi.GPIO
```

## How to Use

### 1. Setup the Hardware

- Connect the Raspberry Pi to the RC car chassis.
- Connect the motors to the GPIO pins as defined in `rpi/stream_video.py`.
- Ensure the Raspberry Pi camera is properly connected.

### 2. Configure Network Settings

Make sure the Raspberry Pi and the server computer are on the same network and can communicate with each other.

1. On the Raspberry Pi, modify `rpi/utils.py` to set the correct server IP address:
```python
server_address = ('SERVER_IP_ADDRESS', 8000)
command_address = ('SERVER_IP_ADDRESS', 8001)
```

### 3. Collect Training Data

1. Start the video stream and command receiver on the Raspberry Pi:
```bash
python rpi/stream_video.py
```

2. Start the data collection script on the server:
```bash
python server/collect_data.py
```

3. Use the arrow keys to drive the car around. The script will save images and corresponding commands.
   - Up: Forward
   - Down: Reverse
   - Left: Turn Left
   - Right: Turn Right
   - Up+Left: Forward Left
   - Up+Right: Forward Right
   - X or Q: Exit

The collected data will be saved in the `data_set` directory.

### 4. Visualize and Analyze the Data

After collecting data, you can visualize it to ensure quality:

```bash
python server/visualize_data.py
```

This will:
- Show sample images from each class
- Demonstrate the preprocessing steps (rotation, cropping, high contrast filtering)
- Display class distribution statistics
- Generate visualization plots saved to the project directory

### 5. Tune Preprocessing Parameters

The system includes an interactive tool to tune preprocessing parameters for optimal line detection:

```bash
python server/visualize_data.py --tune
```

You can also specify a specific image to use for tuning:

```bash
python server/visualize_data.py --tune --image collected_images/frame00001.jpg
```

This opens an interactive GUI with sliders to adjust:
- Lower Threshold: filters out very dark pixels (shadows)
- Upper Threshold: filters out very bright pixels (reflections)
- Thresholding Value: controls the final binary threshold

Press 's' to save your parameter settings, or 'q' to quit without saving.

### 6. Train the Model

After collecting enough data and tuning preprocessing parameters, train the neural network model:

```bash
python server/train_model.py
```

This will:
- Load the collected data
- Apply preprocessing (image rotation, cropping, and band-pass filtering)
- Train a convolutional neural network for 200 epochs
- Save the best model to the `models` directory
- Generate training history plots

The script includes:
- Learning rate scheduling
- Early stopping to prevent overfitting
- Model checkpointing to save the best model
- Detailed evaluation metrics including a confusion matrix

### 7. Autonomous Driving

1. Start the video stream on the Raspberry Pi again:
```bash
python rpi/stream_video.py
```

2. Start the autonomous driving script on the server:
```bash
python server/autonomous_drive.py
```

The car will now drive autonomously based on the trained model. The script:
- Shows both the original camera feed and the processed images
- Applies command smoothing to prevent erratic movements
- Displays prediction confidence for each command
- Press 'Q' to exit autonomous mode

## Image Preprocessing

The system uses specialized image preprocessing for line following:

1. **Image Rotation**: Rotates the image 180 degrees to match the camera orientation.
2. **Upper Half Selection**: Takes only the upper half of the rotated image (which was originally the lower half).
3. **Band-pass Filtering**: Filters out very dark areas (shadows) and very bright areas (reflections).
4. **Contrast Enhancement**: Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for better feature visibility.
5. **Binary Thresholding**: Converts the image to binary to isolate the track line.

These preprocessing steps are shared between training and inference for consistency.

## Tips for Success

1. **Collect Diverse Data**: Drive the car in different conditions and scenarios to create a robust dataset.

2. **Balanced Dataset**: Ensure you have a balanced number of examples for each command (forward, left, right, reverse).

3. **Track Design**: Start with a simple track with clear boundaries and a high-contrast black line.

4. **Lighting Conditions**: Try to maintain consistent lighting during data collection and autonomous driving.

5. **Battery Power**: Ensure the Raspberry Pi and motors have sufficient power. Voltage drops can affect performance.

6. **Preprocessing Tuning**: Use the interactive tuning tool to find optimal parameters for your specific track and lighting conditions. This is especially important if you have varying light conditions or shadows on your track.

## Troubleshooting

- If the car is not responding to commands, check the GPIO connections.
- If the video stream is not working, check the network configuration and camera connection.
- If the model is not training properly, ensure you have enough diverse data.
- If line detection is poor, use the parameter tuning tool to adjust the band-pass filter and threshold settings.
- If the car moves erratically, try increasing the command buffer size for smoother transitions. 