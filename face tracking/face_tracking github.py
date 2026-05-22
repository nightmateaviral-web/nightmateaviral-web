from picamera2 import Picamera2
import cv2

"""
Real-Time Face Tracking Visualization
-------------------------------------
Displays live camera feed with
face detection overlays.

Features:
- Real-time video processing
- Haar Cascade face detection
- Visual tracking feedback
- Raspberry Pi optimized

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
# FACE DETECTION SETUP
# ==================================================

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

print("Visual face tracking started.")

# ==================================================
# MAIN LOOP
# ==================================================

while True:

    # Capture frame
    frame = picam2.capture_array()

    # Convert to grayscale
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

    # Draw tracking rectangles
    for (x, y, width, height) in detected_faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        print(f"Face detected at X:{x} Y:{y}")

    # Display camera feed
    cv2.imshow(
        "Face Tracking System",
        frame
    )

    # Exit with ESC key
    if cv2.waitKey(1) == 27:
        break

# ==================================================
# CLEANUP
# ==================================================

cv2.destroyAllWindows()
