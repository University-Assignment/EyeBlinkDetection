import socket
import numpy as np
import cv2
from queue import Queue
from _thread import *

enclosure_queue = Queue()

def webcam(queue):
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()

        if ret == False:
            continue

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)

        data = np.array(imgencode)
        stringData = data.tostring()

        queue.put(stringData)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def threaded(client_socket, queue):

    while True:

        try:
            data = client_socket.recv(1024)
            if not data:
                print('Disconnected')
                break

            stringData = queue.get()
            client_socket.send(str(len(stringData)).ljust(16).encode())
            client_socket.send(stringData)

        except ConnectionResetError as e:

            print('Disconnected')
            break

    client_socket.close()

HOST = '127.0.0.1'
PORT = 9999

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client_socket.connect((HOST, PORT))

start_new_thread(webcam, (enclosure_queue, ))

while True:

    try:
        data = client_socket.recv(1024)
        if not data:
            print('Disconnected')
            break

        stringData = enclosure_queue.get()
        client_socket.send(str(len(stringData)).ljust(16).encode())
        client_socket.send(stringData)

    except ConnectionResetError as e:

        print('Disconnected')
        break

client_socket.close()
