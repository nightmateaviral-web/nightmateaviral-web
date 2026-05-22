from picamera2 import Picamera2
import cv2

"""
Lightweight Face Tracking System
--------------------------------
Optimized for Raspberry Pi Zero 2W.

Features:
- Real-time face detection
- Low-overhead processing
- Terminal-based coordinate output
- Designed for robotics integration

Author: Aviral
"""

# ==================================================
# CAMERA CONFIGURATION
# ==================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

picam2 = Picamera2()

picam2.configure(
    picam2.create_preview_configuration(
        main={
            "format": "XRGB8888",
            "size": (FRAME_WIDTH, FRAME_HEIGHT)
        }
    )
)

picam2.start()

# ==================================================
# FACE DETECTOR
# ==================================================

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

print("Face tracking system initialized.")

# ==================================================
# MAIN LOOP
# ==================================================

while True:

    # Capture camera frame
    frame = picam2.capture_array()

    # Convert image to grayscale
    grayscale_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    detected_faces = face_detector.detectMultiScale(
        grayscale_frame,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # Output face coordinates
    for (x, y, width, height) in detected_faces:

        face_center_x = x + width // 2
        face_center_y = y + height // 2

        print(
            f"Face detected at "
            f"X:{face_center_x} "
            f"Y:{face_center_y}"
        )
