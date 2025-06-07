import socket
import struct
import mss
import cv2
import numpy as np
import time

def start_screen_streaming_client(server_host, server_port=1414):
    # Setup client socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to server at {server_host}:{server_port}...")
    
    try:
        client_socket.connect((server_host, server_port))
        print("Connected to server!")
    except socket.error as e:
        print(f"Connection failed: {e}")
        return
    
    # Initialize screen capture
    with mss.mss() as sct:
        # Get primary monitor
        monitor = sct.monitors[1]
        
        frame_count = 0
        try:
            while True:
                # Capture screen (very fast with mss)
                img = sct.grab(monitor)
                
                # Convert to numpy array
                frame = np.array(img)
                
                # Convert BGRA to BGR (OpenCV format)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Compress frame
                _, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_data = encoded_frame.tobytes()
                
                # Send frame size followed by data
                frame_size = len(frame_data)
                header = struct.pack("!4sI", b"IMGP", frame_size)
                client_socket.sendall(header + frame_data)
                
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Sent {frame_count} frames")
                
                # Limit frame rate
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            client_socket.close()

start_screen_streaming_client('192.168.68.101',server_port=1414)
