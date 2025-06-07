#!/usr/bin/env python3
import socket, struct, time, pickle, zlib
import cv2, mss, numpy as np

SERVER_HOST = '192.168.68.101'
SERVER_PORT = 1414
FRAME_W, FRAME_H = 1280, 720
KEYFRAME_INTERVAL = 60          # send a fresh full frame every N frames
JPEG_Q = 90                     # visually loss-less

def encode_png(img):
    _, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return buf.tobytes()

def encode_jpg(img, q=JPEG_Q):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()

def start_client():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f'Connecting to {SERVER_HOST}:{SERVER_PORT} …')
    sock.connect((SERVER_HOST, SERVER_PORT))
    print('Connected.')

    prev_gray, frame_id = None, 0
    with mss.mss() as sct:
        monitor = sct.monitors[1]

        while True:
            t0 = time.time()

            raw = np.array(sct.grab(monitor))
            frame = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR),
                               (FRAME_W, FRAME_H))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # decide: key-frame?
            if prev_gray is None or frame_id % KEYFRAME_INTERVAL == 0:
                payload = encode_jpg(frame)
                header = struct.pack('!4sIB', b'IMGK', len(payload), 1)
                sock.sendall(header + payload)
                print(f'Sent key-frame {frame_id}')
            else:
                diff = cv2.absdiff(gray, prev_gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                regions = []
                for c in contours:
                    if cv2.contourArea(c) < 100:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    x, y = max(0, x-5), max(0, y-5)
                    w, h = min(FRAME_W-x, w+10), min(FRAME_H-y, h+10)
                    block = encode_png(frame[y:y+h, x:x+w])
                    regions.append({'x': x, 'y': y, 'w': w, 'h': h, 'data': block})

                if regions:
                    delta = {'frame_id': frame_id,
                             'timestamp': time.time(),
                             'regions': regions}
                    blob = zlib.compress(pickle.dumps(delta))
                    header = struct.pack('!4sIB', b'IMGD', len(blob), 0)
                    sock.sendall(header + blob)
                    print(f'Sent delta {frame_id} ({len(regions)} regions)')
                else:
                    print(f'Frame {frame_id}: no change')

            prev_gray, frame_id = gray, frame_id + 1

            # keep 1 FPS
            dt = time.time() - t0
            if dt < 1:
                time.sleep(1 - dt)

if __name__ == '__main__':
    start_client()
