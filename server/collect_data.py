"""collect_data.py: Collects the training images and classifies them based
on the user car control"""

import cv2
import numpy as np
import socket
import struct
import pygame
from pygame.locals import *
import time
import os
import sys
from image_preprocessing import ImagePreprocessor


class CollectData(object):
    def __init__(self):
        # Create directories for saved data and images
        if not os.path.exists('data_set'):
            os.makedirs('data_set')
        if not os.path.exists('collected_images'):
            os.makedirs('collected_images')
            
        # Initialize image preprocessor
        self.preprocessor = ImagePreprocessor()
            
        # Videostream (kamera från Raspberry Pi)
        self.sock = socket.socket()
        self.sock.bind(('0.0.0.0', 8000))  # För video
        self.sock.listen(1)
        print("Waiting for Raspberry Pi video connection...")
        self.connection = self.sock.accept()[0].makefile('rb')
        print("Raspberry Pi video connected.")

        # Kontrollsocket – Raspberry Pi ansluter hit
        self.cmd_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.cmd_server_socket.bind(('0.0.0.0', 8001))  # Lyssna på kommandon
        self.cmd_server_socket.listen(1)
        print("Waiting for Raspberry Pi control connection...")
        self.cmd_connection, _ = self.cmd_server_socket.accept()
        print("Raspberry Pi control connected.")

        self.send_instr = True

        # Klassetiketter (vänster, höger, framåt, bakåt)
        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1

        # Initiera pygame
        pygame.init()
        pygame.display.set_mode((400, 300))

        self.collect_images()

    def collect_images(self):
        saved_frames = 0
        total_frames = 0
        start_time = cv2.getTickCount()

        print('Starting data collection...')
        # Calculate the size of the full image for the flattened array
        full_image_size = 240*320
        image_array = np.zeros((1, full_image_size), 'float')
        label_array = np.zeros((1, 4), 'float')

        try:
            frame = 1
            complex_cmd = False

            while self.send_instr:
                # Läs en bildruta
                image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
                if not image_len:
                    break

                recv_bytes = b''
                recv_bytes += self.connection.read(image_len)
                image = cv2.imdecode(np.frombuffer(recv_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

                # Get ROI for display purposes only
                roi = self.preprocessor.preprocess(image, output_type='default')
                
                # Create display with original and processed images
                # Create simple display with labels
                display = np.zeros((roi.shape[0] + 30, roi.shape[1] * 2, 3), dtype=np.uint8)
                
                # Get visualization from preprocessor
                vis_display = self.preprocessor.preprocess(image, output_type='display')
                
                # Show the visualization
                cv2.imshow('Processing Steps', vis_display)
                
                # Convert roi to BGR for main display
                roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                
                # Add to display (just showing the processed ROI now)
                display[30:, :roi.shape[1]] = roi_bgr
                
                # Add labels
                cv2.putText(display, "Processed ROI (display only)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Display
                cv2.imshow('ROI View', display)
                
                # Flatten the full image for training data
                # Normalize the image to float values between 0 and 1
                temp_array = image.reshape(1, full_image_size).astype(np.float32) / 255.0

                frame += 1
                total_frames += 1

                for event in pygame.event.get():
                    if event.type == KEYDOWN or complex_cmd:
                        key = pygame.key.get_pressed()
                        complex_cmd = False
                        cv2.imwrite('collected_images/frame{:>05}.jpg'.format(frame), image)

                        if key[pygame.K_UP] and key[pygame.K_RIGHT]:
                            print("Forward Right")
                            complex_cmd = True
                            self.cmd_connection.send(b'5')
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[1]))
                            saved_frames += 1

                        elif key[pygame.K_UP] and key[pygame.K_LEFT]:
                            print("Forward Left")
                            complex_cmd = True
                            self.cmd_connection.send(b'6')
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[0]))
                            saved_frames += 1

                        elif key[pygame.K_DOWN] and key[pygame.K_RIGHT]:
                            print("Reverse Right")
                            complex_cmd = True
                            self.cmd_connection.send(b'7')

                        elif key[pygame.K_DOWN] and key[pygame.K_LEFT]:
                            print("Reverse Left")
                            complex_cmd = True
                            self.cmd_connection.send(b'8')

                        elif key[pygame.K_UP]:
                            print("Forward")
                            self.cmd_connection.send(b'1')
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[2]))
                            saved_frames += 1

                        elif key[pygame.K_DOWN]:
                            print("Reverse")
                            self.cmd_connection.send(b'2')
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[3]))
                            saved_frames += 1

                        elif key[pygame.K_RIGHT]:
                            print("Right")
                            self.cmd_connection.send(b'3')
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[1]))
                            saved_frames += 1

                        elif key[pygame.K_LEFT]:
                            print("Left")
                            self.cmd_connection.send(b'4')
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[0]))
                            saved_frames += 1

                        elif key[pygame.K_x] or key[pygame.K_q]:
                            print("Exit")
                            self.cmd_connection.send(b'0')
                            self.send_instr = False
                            break

                        else:
                            self.cmd_connection.send(b'0')

                    elif event.type == KEYUP:
                        complex_cmd = False
                        self.cmd_connection.send(b'0')

            # Spara data
            image_array = image_array[1:, :]
            label_array = label_array[1:, :]

            file_name = str(int(time.time()))
            directory = "data_set"
            if not os.path.exists(directory):
                os.makedirs(directory)

            try:
                np.savez(directory + '/' + file_name + '.npz', images=image_array, labels=label_array)
            except IOError as e:
                print(e)

            end_time = cv2.getTickCount()
            duration = (end_time - start_time) / cv2.getTickFrequency()
            print("Streaming duration: {0} seconds".format(duration))
            print(f"Image array shape: {image_array.shape}")
            print(f"Label array shape: {label_array.shape}")
            print(f"Total frames: {total_frames}")
            print(f"Saved frames: {saved_frames}")
            print(f"Dropped frames: {total_frames - saved_frames}")

        finally:
            self.connection.close()
            self.sock.close()
            self.cmd_connection.close()
            self.cmd_server_socket.close()


if __name__ == '__main__':
    CollectData()
