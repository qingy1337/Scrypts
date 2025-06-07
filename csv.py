# CLIENT (improved performance)
import socket
import cv2
import numpy as np
import struct

def start_screen_streaming_client(host='localhost', port=9999):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    
    data = b""
    payload_size = struct.calcsize("L")
    
    while True:
        # Receive message size
        while len(data) < payload_size:
            data += client_socket.recv(4096)
            
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]
        
        # Receive frame data
        while len(data) < msg_size:
            data += client_socket.recv(4096)
            
        frame_data = data[:msg_size]
        data = data[msg_size:]
        
        # Decode compressed frame
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Display
        cv2.imshow('Screen', frame)
        if cv2.waitKey(1) == ord('q'):
            break
