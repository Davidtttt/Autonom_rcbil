import io
import time
import struct
import socket
import threading
from picamera2 import Picamera2
from utils import server_address, command_address  # Lägg till command_address t.ex. ('192.168.X.X', 8001)
import RPi.GPIO as GPIO

GPIO_RIGHT_A = 25
GPIO_RIGHT_B = 27
GPIO_LEFT_A = 24
GPIO_LEFT_B = 23
GPIO_SPEED = 22

# ========== FUNKTION: Kör kommandomottagare ==========
def listen_for_commands(cmd_socket):
    try:
        while True:
            cmd = cmd_socket.recv(1)
            if not cmd:
                break
            print(f"[COMMAND RECEIVED]: {cmd}")
            if cmd == b'1':
                # Forwards
                GPIO.output(GPIO_LEFT_A, GPIO.HIGH)
                GPIO.output(GPIO_RIGHT_A, GPIO.HIGH)
                GPIO.output(GPIO_LEFT_B, GPIO.LOW)
                GPIO.output(GPIO_RIGHT_B, GPIO.LOW)
                pwm.ChangeDutyCycle(50)
            elif cmd == b'2':
                # Reverse
                GPIO.output(GPIO_LEFT_B, GPIO.HIGH)
                GPIO.output(GPIO_RIGHT_B, GPIO.HIGH)
                GPIO.output(GPIO_LEFT_A, GPIO.LOW)
                GPIO.output(GPIO_RIGHT_A, GPIO.LOW)
                pwm.ChangeDutyCycle(50)
            elif cmd == b'3' or cmd == b'5':
                # Right
                GPIO.output(GPIO_LEFT_A, GPIO.HIGH)
                GPIO.output(GPIO_RIGHT_A, GPIO.LOW)
                GPIO.output(GPIO_LEFT_B, GPIO.LOW)
                GPIO.output(GPIO_RIGHT_B, GPIO.LOW)
                pwm.ChangeDutyCycle(40)
            elif cmd == b'4' or cmd == b'6':
                # Left
                GPIO.output(GPIO_LEFT_A, GPIO.LOW)
                GPIO.output(GPIO_RIGHT_A, GPIO.HIGH)
                GPIO.output(GPIO_LEFT_B, GPIO.LOW)
                GPIO.output(GPIO_RIGHT_B, GPIO.LOW)
                pwm.ChangeDutyCycle(40)
            elif cmd == b'0':
                # STop
                GPIO.output(GPIO_LEFT_A, GPIO.LOW)
                GPIO.output(GPIO_RIGHT_A, GPIO.LOW)
                GPIO.output(GPIO_LEFT_B, GPIO.LOW)
                GPIO.output(GPIO_RIGHT_B, GPIO.LOW)
                pwm.ChangeDutyCycle(0)
            # Här lägger du till styrlogik, t.ex.:
            # if cmd == b'1':
            #     motor_controller.forward()
            # elif cmd == b'0':
            #     motor_controller.stop()
            # etc...
    except Exception as e:
        print(f"Command listener error: {e}")
    finally:
        cmd_socket.close()
        print("Command socket closed.")

# Function to stop motors
def stop_motors():
    GPIO.output(GPIO_LEFT_A, GPIO.LOW)
    GPIO.output(GPIO_RIGHT_A, GPIO.LOW)
    GPIO.output(GPIO_LEFT_B, GPIO.LOW)
    GPIO.output(GPIO_RIGHT_B, GPIO.LOW)
    print("[MOTORS] Motors stopped")

# List of GPIO pins used
gpio_pins = [GPIO_LEFT_A, GPIO_LEFT_B, GPIO_RIGHT_A, GPIO_RIGHT_B]

# Setup GPIO
GPIO.setmode(GPIO.BCM)  # Use BCM numbering
for pin in gpio_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

GPIO.setup(GPIO_SPEED, GPIO.OUT)
pwm = GPIO.PWM(GPIO_SPEED, 100) # 100 Hz
pwm.start(50)  # 50% duty cycle

# ========== VIDEOSTREAM ==========
# Skapa TCP/IP-socket för videoström
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('[VIDEO] Connecting to server...')
client_socket.connect(server_address)
print('[VIDEO] Connected to server')
connection = client_socket.makefile('wb')

# ========== STYRKOMMANDO ==========
cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('[COMMAND] Connecting to command server...')
cmd_socket.connect(command_address)
print('[COMMAND] Connected to command server')

# Starta tråd för kommandomottagning
command_thread = threading.Thread(target=listen_for_commands, args=(cmd_socket,))
command_thread.daemon = True
command_thread.start()

try:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (320, 240)}, buffer_count=1)
    picam2.configure(config)
    picam2.start()

    time.sleep(2)  # ge kameran tid att initieras

    stream = io.BytesIO()
    start_time = time.time()
    duration = 60*3  # sekunder
    
    # Frame rate control - max 10 FPS
    max_fps = 10
    min_frame_time = 1.0 / max_fps
    last_frame_time = time.time()

    while time.time() - start_time < duration:
        # Calculate time since last frame
        current_time = time.time()
        elapsed = current_time - last_frame_time
        
        # If we're running faster than our max FPS, sleep to maintain the rate
        if elapsed < min_frame_time:
            time.sleep(min_frame_time - elapsed)
            
        # Reset for next frame
        last_frame_time = time.time()
        
        stream.seek(0)
        stream.truncate()

        picam2.capture_file(stream, format='jpeg')

        connection.write(struct.pack('<L', stream.tell()))
        connection.flush()

        stream.seek(0)
        connection.write(stream.read())
        connection.flush()

    # Signalera slut
    connection.write(struct.pack('<L', 0))
    connection.flush()

except KeyboardInterrupt:
    print("\n[INTERRUPT] Ctrl+C detected")
    stop_motors()

finally:
    print('[SHUTDOWN] Closing connections...')
    stop_motors()
    connection.close()
    client_socket.close()
    cmd_socket.close()
    GPIO.cleanup()
