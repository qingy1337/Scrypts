import socket
import struct
import mss
import cv2
import numpy as np
import time
import zlib

def start_delta_streaming_client(server_host, server_port=1414):
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
        prev_frame = None
        
        try:
            while True:
                # Capture screen once per second
                start_time = time.time()
                
                # Capture screenshot
                img = sct.grab(monitor)
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Resize to reduce data (optional)
                frame = cv2.resize(frame, (1280, 720))
                
                # Convert to grayscale for change detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is None:
                    # First frame - send as keyframe
                    print("Sending initial keyframe")
                    _, jpg_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_data = jpg_frame.tobytes()
                    
                    # Send as keyframe
                    header = struct.pack("!4sIB", b"IMGK", len(frame_data), 1)  # 1 = keyframe
                    client_socket.sendall(header + frame_data)
                else:
                    # Compute difference with previous frame
                    diff = cv2.absdiff(gray, prev_frame_gray)
                    
                    # Threshold to find areas with significant changes
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    
                    # Find contours of changed regions
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Filter out small changes
                    significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
                    
                    if len(significant_contours) > 0:
                        # Create a mask of changed regions
                        mask = np.zeros_like(thresh)
                        cv2.drawContours(mask, significant_contours, -1, 255, -1)
                        
                        # Find bounding rectangles of changed regions
                        regions = []
                        for contour in significant_contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            # Expand slightly to ensure all changes are captured
                            x = max(0, x - 5)
                            y = max(0, y - 5)
                            w = min(frame.shape[1] - x, w + 10)
                            h = min(frame.shape[0] - y, h + 10)
                            
                            # Extract region
                            region = frame[y:y+h, x:x+w]
                            
                            # Compress region
                            _, jpg_region = cv2.imencode('.jpg', region, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            
                            regions.append({
                                'x': x,
                                'y': y, 
                                'w': w,
                                'h': h,
                                'data': jpg_region.tobytes()
                            })
                        
                        # Prepare delta frame
                        delta_data = {
                            'frame_id': frame_count,
                            'timestamp': time.time(),
                            'regions': regions
                        }
                        
                        # Serialize and compress
                        delta_bytes = bytes(str(delta_data), 'utf-8')
                        compressed_delta = zlib.compress(delta_bytes)
                        
                        # Send delta frame
                        header = struct.pack("!4sIB", b"IMGD", len(compressed_delta), 0)  # 0 = delta frame
                        client_socket.sendall(header + compressed_delta)
                        
                        print(f"Sent delta frame {frame_count} with {len(regions)} changed regions")
                    else:
                        print(f"Frame {frame_count} - No significant changes detected")
                
                # Store current frame for next comparison
                prev_frame = frame
                prev_frame_gray = gray
                
                frame_count += 1
                
                # Wait until 1 second has elapsed
                elapsed = time.time() - start_time
                if elapsed < 1.0:
                    time.sleep(1.0 - elapsed)
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            client_socket.close()

start_delta_streaming_client('192.168.68.101', 1414)
