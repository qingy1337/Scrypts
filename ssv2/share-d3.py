#!/usr/bin/env python3
"""
Screen-to-socket delta streamer (client side)
--------------------------------------------

Improvements over the original:
*   **Colour-space diff** instead of grayscale-only.
*   Lower change threshold (THRESH_DELTA = 8) so “erase” pixels get detected.
*   Sanity-clamp each contour’s bounding box so we never request pixels
    outside the 1280 × 720 buffer.
*   Key-frame every 500 frames and PNG-encoded deltas (loss-less).

Drop this file in place of your old client script and run it with
`python3 client_delta_stream.py`.
"""

import socket, struct, time, pickle, zlib
import cv2, mss, numpy as np

# --------------------------------------------------------------------------- #
#  Config                                                                     #
# --------------------------------------------------------------------------- #
SERVER_HOST = "192.168.68.101"
SERVER_PORT = 1414

FRAME_W, FRAME_H = 1280, 720          # down-sampled resolution to transmit
KEYFRAME_INTERVAL = 500               # send a full frame every N frames
JPEG_Q = 90                           # quality for key-frames
THRESH_DELTA = 8                      # 0–255 – lower = more sensitive
MIN_CONTOUR_AREA = 100                # ignore tiny noise

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def encode_png(img: np.ndarray) -> bytes:
    """Loss-less PNG (good for small changed regions)."""
    _, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return buf.tobytes()

def encode_jpg(img: np.ndarray, q: int = JPEG_Q) -> bytes:
    """Visually loss-less full-frame JPEG."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()

# --------------------------------------------------------------------------- #
#  Main loop                                                                  #
# --------------------------------------------------------------------------- #
def start_client():
    # ---------- socket handshake ----------
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to {SERVER_HOST}:{SERVER_PORT} …")
    sock.connect((SERVER_HOST, SERVER_PORT))
    print("Connected.")

    prev_frame, frame_id = None, 0

    with mss.mss() as sct:
        monitor = sct.monitors[1]          # primary screen

        while True:
            t0 = time.time()

            # ---------- capture & resize ----------
            raw  = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            # ---------- choose key-frame or delta ----------
            if prev_frame is None or frame_id % KEYFRAME_INTERVAL == 0:
                # --- key-frame ---
                payload = encode_jpg(frame)
                header  = struct.pack("!4sIB", b"IMGK", len(payload), 1)
                sock.sendall(header + payload)
                print(f"Sent key-frame {frame_id}")
            else:
                # --- delta frame ---
                colour_diff = cv2.absdiff(frame, prev_frame)
                diff_gray   = cv2.cvtColor(colour_diff, cv2.COLOR_BGR2GRAY)

                _, thresh = cv2.threshold(
                    diff_gray, THRESH_DELTA, 255, cv2.THRESH_BINARY
                )
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                regions = []
                for c in contours:
                    if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                        continue

                    x, y, w, h = cv2.boundingRect(c)
                    # pad a little so we don’t miss edges
                    x, y = max(0, x - 5), max(0, y - 5)
                    w, h = min(FRAME_W - x, w + 10), min(FRAME_H - y, h + 10)

                    block = encode_png(frame[y : y + h, x : x + w])
                    regions.append(
                        {"x": x, "y": y, "w": w, "h": h, "data": block}
                    )

                if regions:
                    delta = {
                        "frame_id": frame_id,
                        "timestamp": time.time(),
                        "regions": regions,
                    }
                    blob   = zlib.compress(pickle.dumps(delta))
                    header = struct.pack("!4sIB", b"IMGD", len(blob), 0)
                    sock.sendall(header + blob)
                    print(f"Sent delta {frame_id}  ({len(regions)} regions)")
                else:
                    print(f"Frame {frame_id}: no visible change")

            # ---------- book-keeping ----------
            prev_frame = frame
            frame_id  += 1

            # 1 fps pacing
            dt = time.time() - t0
            if dt < 1.0:
                time.sleep(1.0 - dt)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    start_client()
