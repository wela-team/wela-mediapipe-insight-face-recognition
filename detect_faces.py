"""Main entry point for face detection."""

import argparse
from typing import Union

from configs.config import IN_CAMERA_INDEX, OUT_CAMERA_INDEX
from core import FaceDetector


def main():
    """Main entry point for face detection."""
    parser = argparse.ArgumentParser(description="Run face detection from camera or video file.")
    parser.add_argument(
        "--camera",
        choices=["in", "out"],
        default="in",
        help="Camera to use: 'in' for built-in, 'out' for external (ignored if --video is set)",
    )
    parser.add_argument(
        "--video",
        metavar="PATH",
        default=None,
        help="Use a video file instead of a camera (e.g. for WSL2 or testing)",
    )
    args = parser.parse_args()

    source: Union[int, str] = IN_CAMERA_INDEX if args.camera == "in" else OUT_CAMERA_INDEX
    if args.video is not None:
        source = args.video
    detector = FaceDetector(camera_index=source, in_out=args.camera)
    detector.start()


if __name__ == "__main__":
    main()
