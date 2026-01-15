# MediaPipe + InsightFace Face Recognition System

A real-time face detection and recognition system using MediaPipe for detection and InsightFace for recognition.

## Project Structure

```
mediapipe-insight-face-recognition/
├── core/                    # Core functionality modules
│   ├── __init__.py
│   ├── detector.py         # FaceDetector class (MediaPipe)
│   ├── recognizer.py       # FaceRecognizer, EmbeddingDatabase, RecognitionProcessor
│   └── memcache_broadcast.py # MemcacheBroadcaster for event broadcasting
│
├── configs/                 # Configuration files
│   ├── __init__.py
│   └── config.py           # All configuration constants
│
├── utils/                   # Utility functions
│   ├── __init__.py
│   └── embedding_builder.py # Build embeddings from sample images
│
├── data/                    # Runtime data (auto-created)
│   ├── temp_faces/         # Temporary face images from detection
│   └── status/             # Status flags for communication
│
├── embeds/                  # Face embeddings storage (auto-created)
│   └── *.npy               # Individual embedding files per person
│
├── faces/                   # Sample face images for building embeddings
│   └── *.jpg, *.png        # Training images (one per person)
│
├── detect_faces.py         # Main entry point for face detection
├── recognize_faces.py       # Main entry point for face recognition
└── requirements.txt        # Python dependencies
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have sample face images in the `faces/` directory (one image per person).

## Usage

### 1. Build Embeddings

First, create embeddings from your sample images:

```bash
python -m utils.embedding_builder
# Or specify custom directory:
python -m utils.embedding_builder faces/ embeds/
```

Or use the function directly:
```python
from utils import build_embeddings
build_embeddings("faces/", "embeds/")
```

### 2. Run Face Detection

In one terminal, start the face detection system:

```bash
python detect_faces.py
```

This will:
- Open your webcam
- Detect faces using MediaPipe
- Save cropped face images to `data/temp_faces/`
- Display detection boxes and recognition status

### 3. Run Face Recognition

In another terminal, start the recognition system:

```bash
python recognize_faces.py
```

This will:
- Monitor `data/temp_faces/` for new images
- Recognize faces using InsightFace
- Compare against known embeddings
- Signal successful recognition via flag file
- Broadcast recognized faces to Memcache (if enabled)

## Configuration

All settings can be modified in `configs/config.py`:

- **Detection settings**: MediaPipe model selection, confidence threshold
- **Recognition settings**: Similarity threshold, InsightFace model
- **Timing constants**: Save intervals, display times
- **Directory paths**: All paths are configurable
- **Memcache settings**: Server address, cooldown period, enable/disable

## Architecture

### Core Modules

- **`core/detector.py`**: `FaceDetector` class handles MediaPipe face detection
- **`core/recognizer.py`**: 
  - `EmbeddingDatabase`: Loads and manages embeddings
  - `FaceRecognizer`: Performs face recognition
  - `RecognitionProcessor`: Processes images from temp directory
- **`core/memcache_broadcast.py`**: `MemcacheBroadcaster` broadcasts recognition events

### Data Flow

```
Webcam → FaceDetector → temp_faces/ → RecognitionProcessor → FaceRecognizer → Flag File
                                                              ↓                ↓
                                                         Embeddings DB    Memcache (if enabled)
```

## Features

- ✅ Real-time face detection with MediaPipe
- ✅ High-accuracy recognition with InsightFace
- ✅ Modular OOP architecture
- ✅ Configurable thresholds and settings
- ✅ Automatic directory management
- ✅ Clean separation of concerns
- ✅ Memcache broadcasting for recognized faces (with cooldown protection)

## Requirements

See `requirements.txt` for full list. Key dependencies:
- opencv-python
- mediapipe
- insightface
- numpy
- scikit-learn
- python-memcached (for Memcache broadcasting)

## Memcache Broadcasting

The system can broadcast recognized faces to a Memcache server for integration with other systems.

### Setup

1. Install and run Memcache server (default: `127.0.0.1:11211`)
2. Configure in `configs/config.py`:
   - `MEMCACHE_SERVER`: Memcache server address
   - `MEMCACHE_COOLDOWN`: Seconds between broadcasts for same person (default: 60)
   - `MEMCACHE_ENABLED`: Set to `False` to disable broadcasting

### How It Works

- When a face is successfully recognized (not "Unknown"), the system broadcasts the recognition event
- The recognized person's image filename is stored in Memcache under the key `id_picture`
- Cooldown protection prevents duplicate broadcasts for the same person within the cooldown period
- The system maps recognized names to image filenames from the `faces/` directory

### Example

When "John Doe" is recognized:
- Memcache key: `id_picture`
- Memcache value: `"John Doe.jpg"` (JSON encoded)
- This prevents duplicate broadcasts for 60 seconds (configurable)
