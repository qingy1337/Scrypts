import socket
import mss
import cv2
import numpy as np
import struct

def start_screen_streaming_server(host='0.0.0.0', port=9999, quality=50):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    
    print(f"Server listening on {host}:{port}")
    client_socket, addr = server_socket.accept()
    print(f"Connection from {addr}")
    
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        
        while True:
            # Fast screen capture with mss
            img = np.array(sct.grab(monitor))
            
            # Convert to BGR (OpenCV format) from BGRA
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Compress with JPEG
            _, encoded_frame = cv2.imencode('.jpg', frame, 
                                           [cv2.IMWRITE_JPEG_QUALITY, quality])
            data = encoded_frame.tobytes()
            
            # Send frame size followed by frame data
            message_size = struct.pack("L", len(data))
            client_socket.sendall(message_size + data)
