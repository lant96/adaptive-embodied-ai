import time

import cv2
import pandas as pd

from adaptive_embodied_ai.acquisition.pose_tracker import PoseTracker
from adaptive_embodied_ai.representation.movement_features import (
    MovementFeatureExtractor,
)


MOVEMENTS = [
    "neutral",
    "raise_arms",
    "lean_left",
    "lean_right",
    "lean_forward",
]

REPETITIONS_PER_MOVEMENT = 5
TRIAL_DURATION = 4.0


def draw_pose_landmarks(frame, pose_landmarks):
    """
    Draw MediaPipe pose landmarks and connections
    using the MediaPipe Tasks API.
    """

    if not pose_landmarks:
        return

    # MediaPipe Tasks returns NormalizedLandmark objects.
    height, width = frame.shape[:2]

    points = []

    for landmark in pose_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        # Draw landmark
        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1,
        )

    # MediaPipe Pose landmark connections.
    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),

        # Face / head
        (9, 10),

        # Upper body
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),

        # Left side
        (11, 23),
        (13, 23),
        (15, 17),
        (15, 19),
        (15, 21),

        # Right side
        (12, 24),
        (14, 24),
        (16, 18),
        (16, 20),
        (16, 22),

        # Torso / hips
        (23, 24),

        # Left leg
        (23, 25),
        (25, 27),
        (27, 29),
        (29, 31),

        # Right leg
        (24, 26),
        (26, 28),
        (28, 30),
        (30, 32),
    ]

    for start_idx, end_idx in connections:
        if (
            start_idx < len(points)
            and end_idx < len(points)
        ):
            cv2.line(
                frame,
                points[start_idx],
                points[end_idx],
                (0, 255, 0),
                2,
            )


def record_trial(
    tracker,
    extractor,
    movement_label,
    participant_id,
    session_id,
    trial_id,
    session_start_time,
):
    extractor.reset()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print()
    print(f"Movement: {movement_label}")
    print(f"Trial: {trial_id}")
    print("Get ready...")

    time.sleep(2)

    print("Recording...")

    rows = []

    trial_start_time = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Failed to read frame from webcam.")
                break

            trial_elapsed = time.perf_counter() - trial_start_time

            if trial_elapsed >= TRIAL_DURATION:
                break

            # Global timestamp for MediaPipe.
            session_elapsed = (
                time.perf_counter() - session_start_time
            )
            timestamp_ms = int(session_elapsed * 1000)

            result = tracker.detect(
                frame,
                timestamp_ms,
            )

            # Draw detected pose landmarks
            if result.pose_landmarks:
                for landmark in result.pose_landmarks[0]:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1,
                    )

            # Extract movement features
            if (
                result.pose_landmarks
                and result.pose_world_landmarks
            ):
                landmarks = result.pose_landmarks[0]
                world_landmarks = result.pose_world_landmarks[0]

                features = extractor.extract(
                    landmarks=landmarks,
                    world_landmarks=world_landmarks,
                    timestamp=trial_elapsed,
                )

                if features is not None:
                    row = {
                        "participant_id": participant_id,
                        "session_id": session_id,
                        "trial_id": trial_id,
                        "movement_label": movement_label,
                        "timestamp": trial_elapsed,
                        **features,
                    }

                    rows.append(row)

            cv2.imshow(
                "Adaptive Embodied AI — Dataset Collection",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Collected {len(rows)} frames.")

    return rows


def main():
    participant_id = input("Participant ID: ").strip()
    session_id = input("Session ID: ").strip()

    if not participant_id:
        raise ValueError(
            "Participant ID cannot be empty."
        )

    if not session_id:
        raise ValueError(
            "Session ID cannot be empty."
        )

    from adaptive_embodied_ai.utils.paths import MOVEMENT_DATA_DIR

    MOVEMENT_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracker = PoseTracker()
    extractor = MovementFeatureExtractor()

    all_rows = []

    trial_id = 1

    session_start_time = time.perf_counter()

    try:
        for movement in MOVEMENTS:
            for repetition in range(
                1,
                REPETITIONS_PER_MOVEMENT + 1,
            ):
                print()
                print("=" * 60)
                print(
                    f"{movement} — "
                    f"repetition {repetition}/"
                    f"{REPETITIONS_PER_MOVEMENT}"
                )
                print("=" * 60)

                rows = record_trial(
                    tracker=tracker,
                    extractor=extractor,
                    movement_label=movement,
                    participant_id=participant_id,
                    session_id=session_id,
                    trial_id=trial_id,
                    session_start_time=session_start_time,
                )

                all_rows.extend(rows)

                trial_id += 1

                print("Trial complete.")

    except KeyboardInterrupt:
        print("\nDataset collection interrupted.")

    if not all_rows:
        raise RuntimeError(
            "No movement data were collected."
        )

    df = pd.DataFrame(all_rows)

    output_path = (
        MOVEMENT_DATA_DIR
        / f"{participant_id}_{session_id}.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 60)
    print("Dataset saved")
    print("=" * 60)
    print(f"File: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Trials: {df['trial_id'].nunique()}")
    print(
        f"Movements: "
        f"{df['movement_label'].nunique()}"
    )


if __name__ == "__main__":
    main()