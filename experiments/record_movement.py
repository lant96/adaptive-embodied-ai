import argparse
import time

import cv2

from adaptive_embodied_ai.acquisition.camera import Camera
from adaptive_embodied_ai.acquisition.pose_tracker import PoseTracker
from adaptive_embodied_ai.acquisition.feature_recorder import FeatureRecorder
from adaptive_embodied_ai.representation.movement_features import (
    MovementFeatureExtractor,
)


DEFAULT_OUTPUT_FILE = "data/movement/session_01.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record movement features from a webcam."
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=(
            "Output CSV path "
            "(default: data/movement/session_01.csv)"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    camera = Camera()
    tracker = PoseTracker()
    extractor = MovementFeatureExtractor()

    extractor.reset()

    print(f"Recording to: {args.output}")
    print("Recording started.")
    print("Press ESC to stop.")

    start_time = time.time()

    try:
        with FeatureRecorder(args.output) as recorder:
            while True:
                frame = camera.read()

                if frame is None:
                    break

                elapsed = time.time() - start_time
                timestamp_ms = int(elapsed * 1000)

                result = tracker.detect(
                    frame,
                    timestamp_ms,
                )

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]

                    features = extractor.extract(
                        landmarks,
                        elapsed,
                    )

                    if features is not None:
                        recorder.record(
                            elapsed,
                            features,
                        )

                cv2.imshow(
                    "Movement Recorder",
                    frame,
                )

                if cv2.waitKey(1) == 27:
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Recording saved to: {args.output}")


if __name__ == "__main__":
    main()