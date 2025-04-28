# Autonomous RC Car with Machine Learning

This project implements an autonomous RC car using machine learning. The system uses a Raspberry Pi with camera for video streaming and motor control, and a server computer for processing video data, training the model, and controlling the car.

## System Overview

The system consists of four main components:

1. **Data Collection** - Collect training data by manually driving the car.
2. **Data Preprocessing** - Process images with rotation, cropping, and advanced line detection techniques.
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

Install the required packages:
```bash
pip install picamera2 RPi.GPIO
```

## Hardware Setup

- Connect the Raspberry Pi to the RC car chassis.
- Connect the motors to the GPIO pins as defined in `rpi/stream_video.py`:
  - GPIO_RIGHT_A = 25
  - GPIO_RIGHT_B = 27
  - GPIO_LEFT_A = 24
  - GPIO_LEFT_B = 23
  - GPIO_SPEED = 22 (PWM pin)
- Ensure the Raspberry Pi camera is properly connected.

## Network Configuration

Make sure the Raspberry Pi and the server computer are on the same network and can communicate with each other.

1. On the Raspberry Pi, modify `rpi/utils.py` to set the correct server IP address:
```python
server_address = ('SERVER_IP_ADDRESS', 8000)
command_address = ('SERVER_IP_ADDRESS', 8001)
```

## Data Collection

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
   - Down+Left: Reverse Left
   - Down+Right: Reverse Right
   - X or Q: Exit

The collected data will be saved in the `data_set` directory as NPZ files, and raw images will be stored in `collected_images`.

## Image Preprocessing

The system uses specialized image preprocessing for line following:

1. **Image Rotation**: Rotates the image 180 degrees to match the camera orientation.
2. **ROI Selection**: Extracts the bottom half of the image where the line is expected.
3. **Advanced Line Detection**: 
   - Multiple thresholding techniques (adaptive, Otsu, and simple)
   - Morphological operations to clean up noise
   - Contour detection to focus on the largest line segments
   - Fallback mechanisms when line detection fails

The `ImagePreprocessor` class in `server/image_preprocessing.py` handles these operations.

## Visualizing Training Data

You can visualize the collected data to ensure quality:

```bash
python server/visualize_training_data.py
```

This will show sample images from each class and demonstrate the preprocessing steps.

## Model Training

After collecting enough data, train the neural network model:

```bash
python server/train_model.py
```

This will:
- Load the collected data from the `data_set` directory
- Apply preprocessing to the images
- Create a balanced dataset across classes
- Train a convolutional neural network with early stopping
- Save the best model to the `server/models` directory as `model_best.h5`
- Generate training history plots

The model architecture includes:
- Multiple convolutional layers with max pooling
- Dropout layers to prevent overfitting
- Dense layers with ReLU activation
- Final softmax layer for classification

## Autonomous Driving

1. Start the video stream on the Raspberry Pi:
```bash
python rpi/stream_video.py
```

2. Start the autonomous driving script on the server:
```bash
python server/autonomous_drive.py
```

The car will now drive autonomously based on the trained model. The script:
- Shows the original camera feed, ROI, and processed images
- Uses command buffering to prevent erratic movements
- Employs failsafe mechanisms for model loading errors
- Press 'Q' to exit autonomous mode

## Project Structure

```
├── collected_images/    # Raw training images
├── data_set/            # Processed training data (NPZ files)
├── logs/                # TensorBoard logs from training
├── rpi/                 # Raspberry Pi code
│   ├── stream_video.py  # Video streaming and motor control
│   ├── takepic.py       # Utility for taking test pictures
│   └── utils.py         # Network configuration
├── server/              # Server-side code
│   ├── autonomous_drive.py     # Autonomous driving script
│   ├── collect_data.py         # Data collection script
│   ├── image_preprocessing.py  # Image preprocessing utilities
│   ├── models/                 # Saved neural network models
│   ├── predict.py              # Prediction utilities
│   ├── train_model.py          # Model training script
│   └── visualize_training_data.py  # Data visualization tool
└── utils/               # Additional utilities
```

## Troubleshooting

- If the car is not responding to commands, check the GPIO connections.
- If the video stream is not working, check the network configuration and camera connection.
- If the model is not training properly, ensure you have enough diverse data.
- If the car moves erratically during autonomous mode, try adjusting the motor speed in `rpi/stream_video.py`.
- If you encounter model loading issues, the system will try alternative models. Check the console output for errors.

## Tips for Success

1. **Collect Diverse Data**: Drive the car in different conditions and scenarios to create a robust dataset.

2. **Balanced Dataset**: Ensure you have a balanced number of examples for each command (forward, left, right, reverse).

3. **Track Design**: Start with a simple track with clear boundaries and a high-contrast line.

4. **Lighting Conditions**: Try to maintain consistent lighting during data collection and autonomous driving.

5. **Battery Power**: Ensure the Raspberry Pi and motors have sufficient power. Voltage drops can affect performance.

6. **Advanced Preprocessing**: The system uses multiple thresholding techniques and contour analysis to handle various lighting conditions.

7. **Command Buffering**: The system uses buffering to smooth out commands and prevent erratic movements.

8. **Model Fallbacks**: If the primary model fails to load, the system will try alternative models.

## Testvideor och demonstration

För att se hur den autonoma RC-bilen presterar i praktiska tester, se följande YouTube-spellista:

[Se testvideor på YouTube](https://www.youtube.com/playlist?list=PLNgUisNxYCkehK0PDd65EGN7uCtbpyE4H)

Spellistan innehåller:
- Körning på hinderbana
- Reaktion på oväntade hinder
- Jämförelse mellan manuell styrning och autonom styrning
