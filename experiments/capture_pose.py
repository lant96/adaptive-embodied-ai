from adaptive_embodied_ai.acquisition.camera import Camera
from adaptive_embodied_ai.acquisition.pose_tracker import PoseTracker

import cv2


camera = Camera()

tracker = PoseTracker()

while True:

    frame = camera.read()

    if frame is None:
        break

    result = tracker.detect(frame)

    if result.pose_landmarks:

        print(
            len(result.pose_landmarks[0])
        )

    cv2.imshow(
        "Adaptive Embodied AI",
        frame,
    )

    if cv2.waitKey(1) == 27:
        break

camera.release()

cv2.destroyAllWindows()