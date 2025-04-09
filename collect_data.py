import io
import time
import struct
import socket
import threading
from picamera2 import Picamera2
from utils import server_address, command_address  # Lägg till command_address t.ex. ('192.168.X.X', 8001)

# ========== FUNKTION: Kör kommandomottagare ==========
def listen_for_commands(cmd_socket):
    try:
        while True:
            cmd = cmd_socket.recv(1)
            if not cmd:
                break
            print(f"[COMMAND RECEIVED]: {cmd}")

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
    duration = 30  # sekunder

    while time.time() - start_time < duration:
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

finally:
    print('[SHUTDOWN] Closing connections...')
    connection.close()
    client_socket.close()
    cmd_socket.close()
