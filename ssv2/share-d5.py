#!/usr/bin/env python3
"""
Delta-streaming client – colour-aware version.
Save as client_delta_stream.py and run with:  python3 client_delta_stream.py
"""

import socket, struct, time, pickle, zlib
import cv2, mss, numpy as np

# --------------------------------------------------------------------------- #
#  Config                                                                     #
# --------------------------------------------------------------------------- #
SERVER_HOST = "192.168.68.101"
SERVER_PORT = 1414

FRAME_W, FRAME_H   = 1280, 720      # transmit resolution
KEYFRAME_INTERVAL  = 600             # send a full frame every N frames
JPEG_Q             = 90             # key-frame quality
THRESH_DELTA       = 4              # ≤ 255 – smaller = more sensitive
MIN_CONTOUR_AREA   = 100            # discard tiny noise
PAD                = 5              # extra pixels around every region

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def encode_png(img: np.ndarray) -> bytes:
    """Loss-less PNG for small regions."""
    _, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return buf.tobytes()

def encode_jpg(img: np.ndarray, q: int = JPEG_Q) -> bytes:
    """Visually loss-less JPEG for key-frames."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()

def clamp(val, min_v, max_v):
    return max(min_v, min(max_v, val))

# --------------------------------------------------------------------------- #
#  Main loop                                                                  #
# --------------------------------------------------------------------------- #
def start_client():
    # 1-. connect
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to {SERVER_HOST}:{SERVER_PORT} …")
    sock.connect((SERVER_HOST, SERVER_PORT))
    print("Connected.")

    prev_frame, frame_id = None, 0

    with mss.mss() as sct:
        monitor = sct.monitors[1]                     # primary screen

        while True:
            t0 = time.time()

            # 2-. capture & resize
            raw   = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            # 3-. choose frame type
            if prev_frame is None or frame_id % KEYFRAME_INTERVAL == 0:
                # --- key-frame ---
                payload = encode_jpg(frame)
                header  = struct.pack("!4sIB", b"IMGK", len(payload), 1)
                sock.sendall(header + payload)
                print(f"Sent key-frame {frame_id}")
            else:
                # --- delta frame ---
                colour_diff = cv2.absdiff(frame, prev_frame)     # 3-channel diff
                diff_max    = colour_diff.max(axis=2).astype(np.uint8)

                _, thresh = cv2.threshold(
                    diff_max, THRESH_DELTA, 255, cv2.THRESH_BINARY
                )
                thresh = cv2.dilate(thresh, None, iterations=1)  # close gaps

                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                regions = []
                for c in contours:
                    if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    x = clamp(x - PAD, 0, FRAME_W)
                    y = clamp(y - PAD, 0, FRAME_H)
                    w = clamp(w + 2 * PAD, 1, FRAME_W - x)
                    h = clamp(h + 2 * PAD, 1, FRAME_H - y)

                    block = encode_png(frame[y : y + h, x : x + w])
                    regions.append(
                        {"x": x, "y": y, "w": w, "h": h, "data": block}
                    )

                if regions:
                    delta  = {"frame_id": frame_id,
                              "timestamp": time.time(),
                              "regions": regions}
                    blob   = zlib.compress(pickle.dumps(delta))
                    header = struct.pack("!4sIB", b"IMGD", len(blob), 0)
                    sock.sendall(header + blob)
                    print(f"Sent delta {frame_id}  ({len(regions)} regions)")
                else:
                    print(f"Frame {frame_id}: no visible change")

            # 4-. prepare for next loop
            prev_frame = frame
            frame_id  += 1

            # keep ~1 FPS
            dt = time.time() - t0
            if dt < 1.0:
                time.sleep(1.0 - dt)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    start_client()
