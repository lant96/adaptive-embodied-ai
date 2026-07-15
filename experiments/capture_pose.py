"""
Simple pose capture experiment.

Purpose:
Test the acquisition pipeline.

Camera
    ↓
MediaPipe
    ↓
Pose landmarks
"""

import cv2

from adaptive_embodied_ai.acquisition.pose_tracker import PoseTracker


def main():

    tracker = PoseTracker()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open camera"
        )

    print("Camera started")

    while True:

        success, frame = camera.read()

        if not success:
            continue


        result = tracker.process(frame)

        landmarks = tracker.extract_landmarks(
            result
        )


        if landmarks:

            print(
                f"Pose detected: "
                f"{len(landmarks)} landmarks"
            )

        else:

            print(
                "No pose detected"
            )


        cv2.imshow(
            "Adaptive Embodied AI - Pose Capture",
            frame
        )


        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break


    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()