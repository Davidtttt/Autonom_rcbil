"""collect_training_data.py: Collects the training images and classifies them based
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


class CollectData(object):
    def __init__(self):
        # Videostream (kamera från Raspberry Pi)
        self.sock = socket.socket()
        self.sock.bind(('0.0.0.0', 8000))  # För video
        self.sock.listen(1)
        self.connection = self.sock.accept()[0].makefile('rb')

        # Kontrollsocket – Raspberry Pi ansluter hit
        self.cmd_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.cmd_server_socket.bind(('0.0.0.0', 8001))  # Lyssna på kommandon
        self.cmd_server_socket.listen(1)
        print("Väntar på anslutning från Raspberry Pi för styrkommandon...")
        self.cmd_connection, _ = self.cmd_server_socket.accept()
        print("Raspberry Pi ansluten för styrkommandon.")

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

        print('Startar datainsamling...')
        image_array = np.zeros((1, 38400), 'float')
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

                # ROI – nedre halvan av bilden
                roi = image[120:240, :]
                cv2.imshow('Video', roi)
                temp_array = roi.reshape(1, 38400).astype(np.float32)

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
            duration = end_time - start_time // cv2.getTickFrequency()
            print("Streaming duration: {0}".format(duration))
            print(image_array.shape)
            print(label_array.shape)
            print("Total frames: {0}".format(total_frames))
            print("Saved frames: {0}".format(saved_frames))
            print("Dropped frames: {0}".format(total_frames - saved_frames))

        finally:
            self.connection.close()
            self.sock.close()
            self.cmd_connection.close()
            self.cmd_server_socket.close()


if __name__ == '__main__':
    CollectData()
