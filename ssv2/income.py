# SERVER (on the computer WITHOUT firewall restrictions)
import socket
import struct
import cv2
import numpy as np

def start_screen_streaming_server(host='0.0.0.0', port=9999):
    # Setup server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    
    print(f"Server listening on {host}:{port}")
    print("Waiting for the source computer to connect...")
    
    client_socket, addr = server_socket.accept()
    print(f"Connection established with {addr}")
    
    data = b""
    payload_size = struct.calcsize("L")
    
    while True:
        # Receive message size
        while len(data) < payload_size:
            packet = client_socket.recv(4096)
            if not packet:
                return  # Connection closed
            data += packet
            
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
        cv2.imshow('Remote Screen', frame)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    start_screen_streaming_server()
