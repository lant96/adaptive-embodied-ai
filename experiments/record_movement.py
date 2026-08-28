import time

import cv2

from adaptive_embodied_ai.acquisition.pose_tracker import PoseTracker
from adaptive_embodied_ai.representation.movement_features import (
    MovementFeatureExtractor,
)


def main():
    tracker = PoseTracker()
    extractor = MovementFeatureExtractor()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Recording movement.")
    print("Press Q to quit.")

    start_time = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Failed to read frame from webcam.")
                break

            elapsed = time.perf_counter() - start_time
            timestamp_ms = int(elapsed * 1000)

            result = tracker.detect(
                frame,
                timestamp_ms,
            )

            if result.pose_landmarks and result.pose_world_landmarks:
                landmarks = result.pose_landmarks[0]
                world_landmarks = result.pose_world_landmarks[0]

                features = extractor.extract(
                    landmarks=landmarks,
                    world_landmarks=world_landmarks,
                    timestamp=elapsed,
                )

                if features is not None:
                    print(features)

            cv2.imshow(
                "Adaptive Embodied AI — Movement Recording",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()