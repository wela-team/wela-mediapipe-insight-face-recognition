"""Main entry point for face recognition."""

import argparse

from core import RecognitionProcessor


def main():
    """Main entry point for face recognition."""
    parser = argparse.ArgumentParser(description="Run face recognition on captured faces.")
    parser.add_argument(
        "--camera",
        choices=["in", "out"],
        default=None,
        help="Direction: 'in' or 'out' (sets direction flag when recognition succeeds)",
    )
    args = parser.parse_args()

    processor = RecognitionProcessor(in_out=args.camera)
    processor.process_loop()


if __name__ == "__main__":
    main()
