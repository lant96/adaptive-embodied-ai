import time

import cv2

from adaptive_embodied_ai.acquisition.camera import Camera
from adaptive_embodied_ai.acquisition.pose_tracker import PoseTracker
from adaptive_embodied_ai.representation.movement_features import (
    MovementFeatureExtractor,
)


def main():
    camera = Camera()
    tracker = PoseTracker()
    extractor = MovementFeatureExtractor()

    extractor.reset()

    print("Feature inspection started.")
    print("Press ESC to stop.")

    start_time = time.time()

    try:
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
                    print(features)

            cv2.imshow(
                "Feature Test",
                frame,
            )

            if cv2.waitKey(1) == 27:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()