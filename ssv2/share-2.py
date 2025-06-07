import socket
import struct
import ctypes
from ctypes import windll
from PIL import Image
import io
import time

def start_screen_streaming_client(server_host, server_port=9999):
    # Setup client socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to server at {server_host}:{server_port}...")
    
    # Connect with retry logic
    connected = False
    while not connected:
        try:
            client_socket.connect((server_host, server_port))
            connected = True
            print("Connected to server!")
        except socket.error:
            print("Connection failed. Retrying in 5 seconds...")
            time.sleep(5)
    
    # Get screen dimensions
    user32 = windll.user32
    width = user32.GetSystemMetrics(0)
    height = user32.GetSystemMetrics(1)
    
    # Setup for screen capture
    SRCCOPY = 0xCC0020
    hdcScreen = windll.user32.GetDC(None)
    hdcMem = windll.gdi32.CreateCompatibleDC(hdcScreen)
    bmp = windll.gdi32.CreateCompatibleBitmap(hdcScreen, width, height)
    windll.gdi32.SelectObject(hdcMem, bmp)
    
    try:
        while True:
            # Direct Windows API screen capture
            windll.gdi32.BitBlt(hdcMem, 0, 0, width, height, hdcScreen, 0, 0, SRCCOPY)
            
            # Convert to image
            bmpinfo = struct.pack('LHHHH', struct.calcsize('LHHHH'), width, height, 1, 32)
            buffer = ctypes.create_string_buffer(width * height * 4)
            windll.gdi32.GetDIBits(hdcMem, bmp, 0, height, buffer, bmpinfo, 0)
            
            # Create image from buffer
            img = Image.frombuffer('RGBA', (width, height), buffer, 'raw', 'BGRA', 0, 1)
            
            # Convert RGBA to RGB before saving as JPEG
            rgb_img = img.convert('RGB')
            
            # Compress with JPEG and send
            with io.BytesIO() as output:
                # Use rgb_img instead of img
                rgb_img.save(output, format='JPEG', quality=70)
                data = output.getvalue()
                
            message_size = struct.pack("L", len(data))
            client_socket.sendall(message_size + data)
            
    except (ConnectionResetError, BrokenPipeError):
        print("Server disconnected")
    finally:
        # Clean up resources
        windll.gdi32.DeleteObject(bmp)
        windll.gdi32.DeleteDC(hdcMem)
        windll.user32.ReleaseDC(None, hdcScreen)
        client_socket.close()

start_screen_streaming_client('192.168.68.101',server_port=1414)
