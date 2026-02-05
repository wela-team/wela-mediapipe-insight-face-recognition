"""Main entry point for face detection."""

import argparse

from configs.config import IN_CAMERA_INDEX, OUT_CAMERA_INDEX
from core import FaceDetector


def main():
    """Main entry point for face detection."""
    parser = argparse.ArgumentParser(description="Run face detection from camera.")
    parser.add_argument(
        "--camera",
        choices=["in", "out"],
        default="in",
        help="Camera to use: 'in' for built-in, 'out' for external",
    )
    args = parser.parse_args()

    camera_index = IN_CAMERA_INDEX if args.camera == "in" else OUT_CAMERA_INDEX
    detector = FaceDetector(camera_index=camera_index)
    detector.start()


if __name__ == "__main__":
    main()
