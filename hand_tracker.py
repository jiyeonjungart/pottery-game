"""
Hand Tracker — index finger tip follower
----------------------------------------
Dependencies:
    pip install opencv-python mediapipe

Model file (auto-downloaded or place manually):
    hand_landmarker.task  (same directory as this script)

Run:
    python3 hand_tracker.py

Press  Q  or  ESC  to quit.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
import os, urllib.request

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task …")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH,
    )
    print("Done.")

# ── MediaPipe Tasks setup ─────────────────────────────────────────────────────
base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options   = HandLandmarkerOptions(
    base_options          = base_opts,
    num_hands             = 1,
    min_hand_detection_confidence = 0.7,
    min_hand_presence_confidence  = 0.6,
    min_tracking_confidence       = 0.6,
)
detector = HandLandmarker.create_from_options(options)

# INDEX_FINGER_TIP is landmark index 8
INDEX_TIP = 8

# Dot appearance
DOT_RADIUS = 14
DOT_COLOR  = (0, 220, 255)   # BGR — vivid yellow-orange
TRAIL      = []
TRAIL_LEN  = 8

# Connection pairs for drawing the hand skeleton (MediaPipe standard)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ── Webcam ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Hand tracker running — press Q or ESC to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # Convert to MediaPipe Image (RGB)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_img)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            # Draw skeleton
            for a, b in HAND_CONNECTIONS:
                x1, y1 = int(hand[a].x * w), int(hand[a].y * h)
                x2, y2 = int(hand[b].x * w), int(hand[b].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (150, 150, 150), 1, cv2.LINE_AA)
            for lm in hand:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (80, 80, 80), -1)

            # Index fingertip
            tip = hand[INDEX_TIP]
            cx, cy = int(tip.x * w), int(tip.y * h)

            TRAIL.append((cx, cy))
            if len(TRAIL) > TRAIL_LEN:
                TRAIL.pop(0)

            # Fading trail
            for i, (tx, ty) in enumerate(TRAIL[:-1]):
                alpha  = (i + 1) / TRAIL_LEN
                radius = max(3, int(DOT_RADIUS * alpha * 0.6))
                color  = tuple(int(c * alpha) for c in DOT_COLOR)
                cv2.circle(frame, (tx, ty), radius, color, -1, cv2.LINE_AA)

            # Main dot
            cv2.circle(frame, (cx, cy), DOT_RADIUS,     DOT_COLOR,       -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), DOT_RADIUS + 2, (255, 255, 255),  2, cv2.LINE_AA)
    else:
        TRAIL.clear()

    cv2.putText(frame, "Index-finger tracker  |  Q / ESC to quit",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("Hand Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
detector.close()
print("Closed.")
