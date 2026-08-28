import time

import cv2

from adaptive_embodied_ai.acquisition.pose_tracker import PoseTracker


def main():
    tracker = PoseTracker()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

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

            display_frame = frame.copy()

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                height, width = display_frame.shape[:2]

                for landmark in landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        display_frame,
                        (x, y),
                        3,
                        (0, 255, 0),
                        -1,
                    )

            cv2.imshow(
                "Adaptive Embodied AI — Pose",
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()