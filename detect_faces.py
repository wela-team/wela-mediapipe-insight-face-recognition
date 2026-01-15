"""Main entry point for face detection."""

from core import FaceDetector


def main():
    """Main entry point for face detection."""
    detector = FaceDetector()
    detector.start()


if __name__ == "__main__":
    main()
