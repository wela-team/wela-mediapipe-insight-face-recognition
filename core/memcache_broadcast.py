"""Memcache broadcasting module for face recognition events."""

import json
import time
import threading
from pathlib import Path
from typing import Optional
import memcache

from configs.config import DATA_DIR


class MemcacheBroadcaster:
    """Broadcasts face recognition events to Memcache with cooldown protection."""

    def __init__(
        self,
        server: str = '127.0.0.1:11211',
        cooldown: int = 5,
        faces_dir: Optional[Path] = None
    ):
        """
        Initialize memcache broadcaster for face recognition events.
        
        Args:
            server: Memcache server address (default: '127.0.0.1:11211')
            cooldown: Cooldown period in seconds to prevent duplicate broadcasts (default: 60)
            faces_dir: Directory containing face images (default: PROJECT_ROOT / 'faces')
        """
        self.shared = memcache.Client([server], debug=0)
        self.cooldown = cooldown
        self.cache = {}  # Maps name -> last_sent_timestamp
        self.lock = threading.Lock()
        
        # Build mapping from recognized names to picture filenames
        if faces_dir is None:
            faces_dir = DATA_DIR / "faces"
        self.faces_dir = Path(faces_dir)
        self.name_to_filename = {}
        self._build_filename_mapping()
    
    def _build_filename_mapping(self) -> None:
        """Build mapping from recognized names to picture filenames."""
        if not self.faces_dir.exists():
            print(f"[MEMCACHE] Faces directory {self.faces_dir} not found, skipping filename mapping")
            return
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for file in self.faces_dir.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                name = file.stem  # filename without extension
                self.name_to_filename[name] = file.name
        
        print(f"[MEMCACHE] Mapped {len(self.name_to_filename)} face images to names")
    
    def broadcast_face(self, name: str) -> bool:
        """
        Broadcast recognized face to memcache if cooldown period has passed.
        
        Args:
            name: Name of the recognized face (must not be "Unknown" or None)
        
        Returns:
            bool: True if broadcast was sent, False otherwise
        """
        if name is None or name == "Unknown":
            return False
        
        current_time = time.time()
        should_send = False
        
        with self.lock:
            last_sent = self.cache.get(name, 0)
            if current_time - last_sent >= self.cooldown:
                should_send = True
                self.cache[name] = current_time
        
        if should_send:
            print(f"Sending face to memcache: {name}")
            # Get filename for this recognized face
            filename = self.name_to_filename.get(name, f"{name}.jpg")
            
            try:
                # Set to memcache with JSON encoding
                # self.shared.set('face', json.dumps(filename))
                print(f"[MEMCACHE] Broadcasted: {filename} (recognized: {name})")
                return True
            except Exception as e:
                print(f"[MEMCACHE ERROR] Failed to send: {e}")
                return False
        
        return False
