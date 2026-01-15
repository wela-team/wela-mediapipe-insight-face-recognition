"""Configuration constants for face detection and recognition system."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
EMBEDS_DIR = DATA_DIR / "embeddings"
TEMP_DIR = DATA_DIR / "temp_faces"
STATUS_DIR = DATA_DIR / "status"
FACES_DIR = DATA_DIR / "faces"

# File paths
FLAG_FILE = STATUS_DIR / "recognized.flag"
PROCESSING_FLAG_FILE = STATUS_DIR / "processing.flag"

# Timing constants
SAVE_INTERVAL = 0.4  # seconds between face saves
GREEN_DISPLAY_TIME = 0.5  # seconds to display green recognition box
RECOGNITION_POLL_INTERVAL = 0.05  # seconds between recognition checks
FLAG_DELETE_DELAY = 0.05  # seconds before deleting flag file

# Detection settings
MEDIAPIPE_MODEL_SELECTION = 0
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.6
FACE_CROP_MARGIN = 0.2  # 20% margin around detected face

# Recognition settings
RECOGNITION_THRESHOLD = 0.45
INSIGHTFACE_MODEL_NAME = "buffalo_l"
INSIGHTFACE_PROVIDERS = ["CPUExecutionProvider"]
MIN_FACE_SIZE = 50  # pixels
MIN_FACE_DIMENSION_FOR_RECOGNITION = 200  # pixels

# Camera settings
CAMERA_INDEX = 0

# Memcache settings
MEMCACHE_SERVER = '127.0.0.1:11211'
MEMCACHE_COOLDOWN = 60  # seconds between broadcasts for same person
MEMCACHE_ENABLED = False  # Set to False to disable memcache broadcasting

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
EMBEDS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
STATUS_DIR.mkdir(exist_ok=True)
FACES_DIR.mkdir(exist_ok=True)
