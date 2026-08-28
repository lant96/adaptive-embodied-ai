import time

import cv2
import pandas as pd

from adaptive_embodied_ai.acquisition.pose_tracker import PoseTracker
from adaptive_embodied_ai.representation.movement_features import (
    MovementFeatureExtractor,
)


DEPTH_FEATURES = [
    "head_offset_z",
    "torso_dz",
    "movement_speed_3d",
]


def main():
    tracker = PoseTracker()
    extractor = MovementFeatureExtractor()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Checking movement features.")
    print("Move naturally in front of the camera.")
    print("Press Q to stop.")

    rows = []

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
                    rows.append(features)

                    print(
                        " | ".join(
                            f"{name}={features[name]:.4f}"
                            for name in DEPTH_FEATURES
                        )
                    )

            cv2.imshow(
                "Adaptive Embodied AI — Feature Check",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not rows:
        raise RuntimeError(
            "No valid pose observations were collected."
        )

    df = pd.DataFrame(rows)

    print()
    print("=" * 60)
    print("Feature summary")
    print("=" * 60)

    print(
        df.describe()[DEPTH_FEATURES].round(4)
    )

    print()
    print("Number of observations:", len(df))

    print()
    print("Missing values:")
    print(df.isna().sum())

    print()
    print("Depth feature ranges:")

    for feature in DEPTH_FEATURES:
        print(
            f"{feature}: "
            f"{df[feature].min():.4f} → "
            f"{df[feature].max():.4f}"
        )


if __name__ == "__main__":
    main()