"""Main entry point for face recognition."""

from core import RecognitionProcessor


def main():
    """Main entry point for face recognition."""
    processor = RecognitionProcessor()
    processor.process_loop()


if __name__ == "__main__":
    main()
