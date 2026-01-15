"""Face recognition module using InsightFace for face matching."""

import os
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

from configs.config import (
    TEMP_DIR,
    STATUS_DIR,
    EMBEDS_DIR,
    FLAG_FILE,
    PROCESSING_FLAG_FILE,
    RECOGNITION_POLL_INTERVAL,
    RECOGNITION_THRESHOLD,
    INSIGHTFACE_MODEL_NAME,
    INSIGHTFACE_PROVIDERS,
    MIN_FACE_SIZE,
    MIN_FACE_DIMENSION_FOR_RECOGNITION,
    FLAG_DELETE_DELAY,
    MEMCACHE_SERVER,
    MEMCACHE_COOLDOWN,
    MEMCACHE_ENABLED,
)


class EmbeddingDatabase:
    """Manages loading and storing face embeddings."""

    def __init__(self, embeddings_dir: Path = EMBEDS_DIR):
        """
        Initialize the embedding database.

        Args:
            embeddings_dir: Directory containing .npy embedding files
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings: Dict[str, list] = {}
        self.load_embeddings()

    def load_embeddings(self) -> None:
        """Load all embeddings from .npy files in the embeddings directory."""
        if not self.embeddings_dir.exists():
            print(f"⚠️ Embeddings directory {self.embeddings_dir} not found!")
            return

        loaded_count = 0
        for npy_file in self.embeddings_dir.glob("*.npy"):
            label = npy_file.stem  # filename without extension
            
            try:
                embedding = np.load(npy_file)
                # Ensure embedding is a numpy array
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                
                # Normalize embedding to unit vector for better cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                # Store as list to maintain compatibility with existing code
                self.embeddings[label] = [embedding]
                loaded_count += 1
                print(f"  ✓ Loaded {label}: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
            except Exception as e:
                print(f"⚠️ Failed to load {npy_file.name}: {e}")

        print(f"✅ Loaded {loaded_count} embeddings from {self.embeddings_dir}")

    def get_all_embeddings(self) -> Dict[str, list]:
        """Get all loaded embeddings."""
        return self.embeddings


class FaceRecognizer:
    """Handles face recognition using InsightFace."""

    def __init__(
        self,
        threshold: float = RECOGNITION_THRESHOLD,
        embeddings_dir: Path = EMBEDS_DIR
    ):
        """
        Initialize the face recognizer.

        Args:
            threshold: Similarity threshold for recognition (0.0 to 1.0)
            embeddings_dir: Directory containing embedding files
        """
        self.threshold = threshold
        self.embedding_db = EmbeddingDatabase(embeddings_dir)
        self.face_analysis = self._initialize_insightface()

    def _initialize_insightface(self) -> FaceAnalysis:
        """Initialize and return InsightFace FaceAnalysis model."""
        app = FaceAnalysis(
            name=INSIGHTFACE_MODEL_NAME,
            providers=INSIGHTFACE_PROVIDERS
        )
        app.prepare(ctx_id=0)
        return app

    def _add_padding(self, face_img: np.ndarray, padding_size: int) -> np.ndarray:
        """
        Add padding around face image for better detection.

        Args:
            face_img: Face image to pad
            padding_size: Size of padding in pixels

        Returns:
            Padded image
        """
        return cv2.copyMakeBorder(
            face_img,
            padding_size, padding_size, padding_size, padding_size,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Black padding
        )

    def _extract_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from cropped face image.

        Args:
            face_img: Cropped face image

        Returns:
            Face embedding vector, or None if extraction fails
        """
        h, w = face_img.shape[:2]
        
        # Check minimum size
        if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
            print(f"⚠️ Face image too small: {w}x{h} (minimum: {MIN_FACE_SIZE})")
            return None

        # Add padding for InsightFace detector
        padding = max(h, w)
        padded_img = self._add_padding(face_img, padding)
        
        # Try to get face embedding
        faces = self.face_analysis.get(padded_img)
        
        if not faces:
            # Fallback: resize small faces and try again
            if h < MIN_FACE_DIMENSION_FOR_RECOGNITION or w < MIN_FACE_DIMENSION_FOR_RECOGNITION:
                scale = max(
                    MIN_FACE_DIMENSION_FOR_RECOGNITION / h,
                    MIN_FACE_DIMENSION_FOR_RECOGNITION / w
                )
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized_face = cv2.resize(face_img, (new_w, new_h))
                
                h_resized, w_resized = resized_face.shape[:2]
                padding = max(h_resized, w_resized)
                padded_img = self._add_padding(resized_face, padding)
                faces = self.face_analysis.get(padded_img)
        
        if not faces:
            print(f"⚠️ InsightFace could not detect face in image ({w}x{h})")
            return None

        embedding = faces[0].embedding
        if embedding is None or len(embedding) == 0:
            print("⚠️ Empty embedding extracted")
            return None
        
        # Ensure embedding is numpy array
        embedding = np.array(embedding)
        
        # Normalize embedding to unit vector for better cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            print("⚠️ Zero-norm embedding extracted")
            return None

        return embedding

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Ensure embeddings are 1D arrays
        emb1 = embedding1.flatten() if embedding1.ndim > 1 else embedding1
        emb2 = embedding2.flatten() if embedding2.ndim > 1 else embedding2
        
        # Check shapes match
        if emb1.shape != emb2.shape:
            print(f"⚠️ Embedding shape mismatch: {emb1.shape} vs {emb2.shape}")
            return 0.0
        
        try:
            score = cosine_similarity(
                emb1.reshape(1, -1),
                emb2.reshape(1, -1)
            )[0][0]
            return float(score)
        except Exception as e:
            print(f"⚠️ Error in similarity calculation: {e}")
            return 0.0

    def recognize(self, face_img: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face from a cropped face image.

        Args:
            face_img: Cropped face image

        Returns:
            Tuple of (name, score). Returns (None, 0.0) if no face detected,
            or ("Unknown", score) if below threshold.
        """
        embedding = self._extract_embedding(face_img)
        if embedding is None:
            return None, 0.0

        # Check if we have any embeddings loaded
        all_embeddings = self.embedding_db.get_all_embeddings()
        if not all_embeddings:
            print("⚠️ No embeddings loaded in database!")
            return "Unknown", 0.0

        best_name = "Unknown"
        best_score = 0.0

        # Compare with all known embeddings
        for name, reference_embeddings in all_embeddings.items():
            for ref_emb in reference_embeddings:
                try:
                    score = self._calculate_similarity(embedding, ref_emb)
                    if score > best_score:
                        best_score = score
                        best_name = name
                except Exception as e:
                    print(f"⚠️ Error calculating similarity for {name}: {e}")
                    continue

        # Debug output
        if best_score == 0.0:
            print(f"⚠️ All similarity scores were 0.0. Embedding shape: {embedding.shape}, "
                  f"Reference embeddings count: {sum(len(embs) for embs in all_embeddings.values())}")

        # Check threshold
        if best_score < self.threshold:
            return "Unknown", best_score

        return best_name, best_score


class RecognitionProcessor:
    """Processes face images from temporary directory and performs recognition."""

    def __init__(self, temp_dir: Path = TEMP_DIR):
        """
        Initialize the recognition processor.

        Args:
            temp_dir: Directory containing temporary face images
        """
        self.temp_dir = Path(temp_dir)
        self.recognizer = FaceRecognizer()
        
        # Initialize memcache broadcaster if enabled
        self.memcache_broadcaster = None
        if MEMCACHE_ENABLED:
            try:
                from .memcache_broadcast import MemcacheBroadcaster
                self.memcache_broadcaster = MemcacheBroadcaster(
                    server=MEMCACHE_SERVER,
                    cooldown=MEMCACHE_COOLDOWN
                )
                print("[MEMCACHE] Broadcaster initialized")
            except ImportError:
                print("[MEMCACHE] Warning: memcache module not installed, broadcasting disabled")
            except Exception as e:
                print(f"[MEMCACHE] Warning: Failed to initialize broadcaster: {e}")

    def _process_image(self, image_path: Path) -> None:
        """
        Process a single face image for recognition.

        Args:
            image_path: Path to the face image file
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return

            name, score = self.recognizer.recognize(img)
            
            if name is None:
                print(f"{image_path.name} → No face detected (image size: {img.shape})")
            else:
                print(f"{image_path.name} → {name} ({score:.2f})")
                self._signal_recognition()
                
                # Broadcast to memcache if valid recognition (not "Unknown")
                if name != "Unknown" and self.memcache_broadcaster is not None:
                    self.memcache_broadcaster.broadcast_face(name)

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")

    def _signal_recognition(self) -> None:
        """Write flag file to signal successful recognition."""
        try:
            FLAG_FILE.write_text("recognized")
        except Exception as e:
            print(f"⚠️ Could not write flag: {e}")

    def _cleanup_image(self, image_path: Path) -> None:
        """
        Clean up processed image file.

        Args:
            image_path: Path to the image file to delete
        """
        try:
            if image_path.exists():
                image_path.unlink()
        except Exception as e:
            print(f"⚠️ Could not delete {image_path}: {e}")

    def _signal_processing(self) -> None:
        """Write flag file to signal that processing is active."""
        try:
            PROCESSING_FLAG_FILE.write_text("processing")
        except Exception as e:
            print(f"⚠️ Could not write processing flag: {e}")

    def _cleanup_processing_flag(self) -> None:
        """Clean up processing flag file."""
        try:
            if PROCESSING_FLAG_FILE.exists():
                PROCESSING_FLAG_FILE.unlink()
        except Exception as e:
            print(f"⚠️ Could not delete processing flag: {e}")

    def _cleanup_flag(self) -> None:
        """Clean up recognition flag file after delay."""
        try:
            if FLAG_FILE.exists():
                time.sleep(FLAG_DELETE_DELAY)
                FLAG_FILE.unlink()
        except Exception as e:
            print(f"⚠️ Could not delete flag: {e}")

    def process_loop(self) -> None:
        """Main processing loop that continuously checks for new face images."""
        print("🟡 Face recognition started...")

        while True:
            try:
                if not self.temp_dir.exists():
                    time.sleep(RECOGNITION_POLL_INTERVAL)
                    continue

                files = list(self.temp_dir.glob("*"))
                files = [f for f in files if f.is_file()]

                if not files:
                    # No files to process, remove processing flag if it exists
                    self._cleanup_processing_flag()
                    time.sleep(RECOGNITION_POLL_INTERVAL)
                    continue

                # Signal that we're processing images
                self._signal_processing()

                for image_path in files:
                    # Process image
                    self._process_image(image_path)
                    
                    # Cleanup
                    self._cleanup_image(image_path)
                    self._cleanup_flag()

                # Cleanup processing flag after all images are processed
                self._cleanup_processing_flag()

            except KeyboardInterrupt:
                print("\n🛑 Recognition stopped by user")
                self._cleanup_processing_flag()
                break
            except Exception as e:
                print(f"❌ Unexpected error in processing loop: {e}")
                self._cleanup_processing_flag()
                time.sleep(RECOGNITION_POLL_INTERVAL)
