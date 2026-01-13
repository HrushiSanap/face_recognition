import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_IMG_DIR = os.path.join(DATA_DIR, "raw")
MODELS_DIR = os.path.join(DATA_DIR, "models")
ENCODINGS_PATH = os.path.join(MODELS_DIR, "encodings.pickle")

# --- Camera & Capture ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TOTAL_PHOTOS_TO_CAPTURE = 25

# --- Recognition & Liveness ---
FRAME_RESIZE_SCALE = 0.25    # Resize frame to 1/4 for speed
RECOGNITION_INTERVAL = 5     # Run recognition every N frames
EYE_AR_THRESH = 0.24         # Threshold: Below this, eyes are "closed"
EYE_AR_CONSEC_FRAMES = 2     # Consecutive frames for a blink
MOVEMENT_THRESHOLD = 2.5     # Minimum pixels nose must move to count as "motion"
LIVENESS_TIMEOUT = 2.0       # Seconds "Live" status lasts after blink