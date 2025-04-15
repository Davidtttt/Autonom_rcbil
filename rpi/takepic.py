from picamera2 import Picamera2
import time

# Initialize the camera
picam2 = Picamera2()

# Configure the camera
picam2.configure(picam2.create_still_configuration())

# Start the camera
picam2.start()
time.sleep(2)  # Give it a moment to adjust to lighting

# Capture and save the image
picam2.capture_file("image.jpg")

# Optional: Stop the camera
picam2.stop()
