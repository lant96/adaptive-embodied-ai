import argparse
import time
from pathlib import Path

import cv2

from adaptive_embodied_ai.acquisition.camera import Camera
from adaptive_embodied_ai.acquisition.feature_recorder import FeatureRecorder
from adaptive_embodied_ai.acquisition.pose_tracker import PoseTracker
from adaptive_embodied_ai.representation.movement_features import (
    MovementFeatureExtractor,
)


MOVEMENTS = [
    ("neutral", "Stay still in a comfortable position."),
    ("raise_arms", "Raise both arms above your head."),
    ("lean_left", "Lean your upper body to the left."),
    ("lean_right", "Lean your upper body to the right."),
    ("lean_forward", "Lean your upper body forward."),
]

DEFAULT_REPETITIONS = 5
DEFAULT_DURATION = 4.0
DEFAULT_REST = 2.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect labelled movement trials."
    )

    parser.add_argument(
        "--participant",
        required=True,
        help="Participant identifier, e.g. P01",
    )

    parser.add_argument(
        "--session",
        required=True,
        help="Session identifier, e.g. session_01",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help=(
            "Number of repetitions per movement "
            f"(default: {DEFAULT_REPETITIONS})"
        ),
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help=(
            "Recording duration per trial in seconds "
            f"(default: {DEFAULT_DURATION})"
        ),
    )

    parser.add_argument(
        "--rest",
        type=float,
        default=DEFAULT_REST,
        help=(
            "Rest duration between trials in seconds "
            f"(default: {DEFAULT_REST})"
        ),
    )

    return parser.parse_args()


def draw_text(frame, lines):
    """Draw multiple lines of information on the camera frame."""

    y = 35

    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        y += 35


def show_instruction(
    camera,
    movement_label,
    instruction,
    trial_number,
):
    """
    Show the movement instruction until SPACE is pressed.

    Returns False if ESC is pressed.
    """

    while True:
        frame = camera.read()

        if frame is None:
            return False

        draw_text(
            frame,
            [
                f"Trial {trial_number}",
                f"Movement: {movement_label}",
                instruction,
                "",
                "Press SPACE to start",
                "Press ESC to stop",
            ],
        )

        cv2.imshow(
            "Movement Dataset Collection",
            frame,
        )

        key = cv2.waitKey(50) & 0xFF

        if key == ord(" "):
            return True

        if key == 27:
            return False


def countdown(camera, seconds=3):
    """Show a short countdown before recording."""

    for remaining in range(seconds, 0, -1):
        start = time.time()

        while time.time() - start < 1.0:
            frame = camera.read()

            if frame is None:
                return False

            draw_text(
                frame,
                [
                    "Get ready...",
                    f"Starting in {remaining}",
                ],
            )

            cv2.imshow(
                "Movement Dataset Collection",
                frame,
            )

            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                return False

    return True


def record_trial(
    camera,
    tracker,
    extractor,
    recorder,
    participant_id,
    session_id,
    trial_id,
    movement_label,
    duration,
    session_start_time,
):
    """
    Record one labelled movement trial.

    MediaPipe receives a session-level timestamp so that timestamps
    remain monotonically increasing across all trials.

    The saved CSV timestamp remains relative to the current trial.
    """

    extractor.reset()

    trial_start_time = time.time()

    while True:
        frame = camera.read()

        if frame is None:
            return False

        trial_elapsed = time.time() - trial_start_time
        session_elapsed = time.time() - session_start_time

        if trial_elapsed >= duration:
            break

        # MediaPipe requires monotonically increasing timestamps
        # across the lifetime of the video detector.
        timestamp_ms = int(session_elapsed * 1000)

        result = tracker.detect(
            frame,
            timestamp_ms,
        )

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            features = extractor.extract(
                landmarks,
                trial_elapsed,
            )

            if features is not None:
                recorder.record(
                    trial_elapsed,
                    {
                        "participant_id": participant_id,
                        "session_id": session_id,
                        "trial_id": trial_id,
                        "movement_label": movement_label,
                        **features,
                    },
                )

        draw_text(
            frame,
            [
                f"Recording: {movement_label}",
                f"Time: {trial_elapsed:.1f} / {duration:.1f}s",
                "Press ESC to stop",
            ],
        )

        cv2.imshow(
            "Movement Dataset Collection",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return False

    return True


def rest_period(camera, seconds):
    """Provide a short rest period between trials."""

    start = time.time()

    while time.time() - start < seconds:
        frame = camera.read()

        if frame is None:
            return False

        remaining = seconds - (time.time() - start)

        draw_text(
            frame,
            [
                "Rest",
                f"Next trial in {max(0, remaining):.1f}s",
            ],
        )

        cv2.imshow(
            "Movement Dataset Collection",
            frame,
        )

        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            return False

    return True


def main():
    args = parse_args()

    if args.repetitions <= 0:
        raise ValueError(
            "Repetitions must be greater than zero."
        )

    if args.duration <= 0:
        raise ValueError(
            "Duration must be greater than zero."
        )

    if args.rest < 0:
        raise ValueError(
            "Rest duration cannot be negative."
        )

    output_directory = Path("data") / "movement"
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_directory
        / f"{args.participant}_{args.session}.csv"
    )

    if output_file.exists():
        raise FileExistsError(
            f"\nDataset already exists:\n"
            f"  {output_file}\n\n"
            "The collector will not overwrite existing data. "
            "Rename or remove the existing file if you intend "
            "to record this session again."
        )

    camera = Camera()
    tracker = PoseTracker()
    extractor = MovementFeatureExtractor()

    trial_number = 0

    # MediaPipe's video detector requires timestamps that continuously
    # increase throughout the lifetime of the tracker.
    session_start_time = time.time()

    try:
        print()
        print("=" * 60)
        print("Movement Dataset Collection")
        print("=" * 60)
        print(f"Participant : {args.participant}")
        print(f"Session     : {args.session}")
        print(f"Repetitions : {args.repetitions}")
        print(f"Duration    : {args.duration}s")
        print(f"Rest        : {args.rest}s")
        print(f"Output      : {output_file}")
        print("=" * 60)
        print()
        print(
            "Follow the instructions shown in the camera window."
        )
        print("Press SPACE to begin each trial.")
        print("Press ESC at any time to stop.")
        print()

        with FeatureRecorder(output_file) as recorder:

            for movement_label, instruction in MOVEMENTS:

                for repetition in range(
                    1,
                    args.repetitions + 1,
                ):
                    trial_number += 1

                    trial_id = f"{trial_number:03d}"

                    print(
                        f"Preparing trial {trial_id}: "
                        f"{movement_label} "
                        f"(repetition {repetition}/"
                        f"{args.repetitions})"
                    )

                    started = show_instruction(
                        camera,
                        movement_label,
                        instruction,
                        trial_number,
                    )

                    if not started:
                        print("Collection stopped.")
                        return

                    if not countdown(camera):
                        print("Collection stopped.")
                        return

                    print(
                        f"Recording trial {trial_id}..."
                    )

                    success = record_trial(
                        camera=camera,
                        tracker=tracker,
                        extractor=extractor,
                        recorder=recorder,
                        participant_id=args.participant,
                        session_id=args.session,
                        trial_id=trial_id,
                        movement_label=movement_label,
                        duration=args.duration,
                        session_start_time=session_start_time,
                    )

                    if not success:
                        print("Collection stopped.")
                        return

                    print(
                        f"Trial {trial_id} completed."
                    )

                    if not rest_period(
                        camera,
                        args.rest,
                    ):
                        print("Collection stopped.")
                        return

        print()
        print("=" * 60)
        print("Dataset collection completed.")
        print(f"Trials recorded: {trial_number}")
        print(f"Saved to: {output_file}")
        print("=" * 60)

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()