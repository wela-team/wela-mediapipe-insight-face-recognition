"""Face detection module using MediaPipe for real-time face detection."""

import cv2
import mediapipe as mp
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Any

from configs.config import (
    TEMP_DIR,
    STATUS_DIR,
    FLAG_FILE,
    SAVE_INTERVAL,
    GREEN_DISPLAY_TIME,
    MEDIAPIPE_MODEL_SELECTION,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    FACE_CROP_MARGIN,
    CAMERA_INDEX,
)


class FaceDetector:
    """Handles face detection using MediaPipe and manages face image saving."""

    def __init__(self, camera_index: int = CAMERA_INDEX):
        """
        Initialize the face detector.

        Args:
            camera_index: Index of the camera to use (default: 0)
        """
        self.camera_index = camera_index
        self.detector = self._initialize_detector()
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Timing state
        self.last_save_time = 0.0
        self.last_green_time = 0.0
        self.recognized_flag_seen = False

    def _initialize_detector(self) -> mp.solutions.face_detection.FaceDetection:
        """Initialize and return MediaPipe face detection model."""
        mp_face = mp.solutions.face_detection
        return mp_face.FaceDetection(
            model_selection=MEDIAPIPE_MODEL_SELECTION,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE
        )

    def _extract_face_region(
        self, frame: Any, detection_box: Any
    ) -> Optional[Any]:
        """
        Extract face region from frame with margin.

        Args:
            frame: Input video frame
            detection_box: MediaPipe detection box

        Returns:
            Cropped face image with margin, or None if extraction fails
        """
        h, w = frame.shape[:2]
        box = detection_box.location_data.relative_bounding_box
        
        # Convert relative coordinates to absolute
        x = int(box.xmin * w)
        y = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)

        # Add margin for better recognition
        x_margin = int(bw * FACE_CROP_MARGIN)
        y_margin = int(bh * FACE_CROP_MARGIN)
        
        # Ensure coordinates are within frame bounds
        x_start = max(0, x - x_margin)
        y_start = max(0, y - y_margin)
        x_end = min(w, x + bw + x_margin)
        y_end = min(h, y + bh + y_margin)

        face = frame[y_start:y_end, x_start:x_end]
        return face if face.size > 0 else None

    def _save_face_image(self, face_img: Any) -> Optional[Path]:
        """
        Save face image to temporary directory.

        Args:
            face_img: Face image to save

        Returns:
            Path to saved file, or None if save failed
        """
        current_time = time.time()
        if current_time - self.last_save_time < SAVE_INTERVAL:
            return None

        filename = TEMP_DIR / f"{uuid.uuid4().hex}.jpg"
        try:
            cv2.imwrite(str(filename), face_img)
            self.last_save_time = current_time
            return filename
        except Exception as e:
            print(f"⚠️ Failed to save face image: {e}")
            return None

    def _draw_detection_box(
        self, frame: Any, detection_box: Any
    ) -> None:
        """
        Draw detection box on frame.

        Args:
            frame: Frame to draw on
            detection_box: MediaPipe detection box
        """
        h, w = frame.shape[:2]
        box = detection_box.location_data.relative_bounding_box
        x = int(box.xmin * w)
        y = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)
        
        # Draw blue detection box
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

    def _update_recognition_status(self, current_time: float) -> None:
        """
        Update recognition status based on flag file.

        Args:
            current_time: Current timestamp
        """
        if FLAG_FILE.exists():
            if not self.recognized_flag_seen:
                self.last_green_time = current_time
                self.recognized_flag_seen = True
        else:
            self.recognized_flag_seen = False

    def _draw_recognition_indicator(self, frame: Any, current_time: float) -> None:
        """
        Draw green recognition indicator if face was recently recognized.

        Args:
            frame: Frame to draw on
            current_time: Current timestamp
        """
        if current_time - self.last_green_time < GREEN_DISPLAY_TIME:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (20, 20), (w - 20, h - 20), (0, 255, 0), 4)
            cv2.putText(
                frame,
                "FACE RECOGNIZED",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

    def _process_detections(self, frame: Any, detections: Any) -> None:
        """
        Process face detections: extract, save, and draw boxes.

        Args:
            frame: Current video frame
            detections: MediaPipe detection results
        """
        if not detections:
            return

        for detection in detections:
            face_img = self._extract_face_region(frame, detection)
            if face_img is not None:
                self._save_face_image(face_img)
                self._draw_detection_box(frame, detection)

    def start(self) -> None:
        """Start the face detection loop."""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")

        print("🟢 Face detection started...")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.detector.process(rgb_frame)

                current_time = time.time()

                # Process detections
                self._process_detections(frame, results.detections)

                # Update recognition status
                self._update_recognition_status(current_time)

                # Draw recognition indicator
                self._draw_recognition_indicator(frame, current_time)

                # Display frame
                cv2.imshow("Detection", frame)

                # Exit on ESC key
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
