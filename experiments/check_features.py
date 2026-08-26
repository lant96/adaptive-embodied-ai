from adaptive_embodied_ai.acquisition.camera import Camera
from adaptive_embodied_ai.acquisition.pose_tracker import PoseTracker

from adaptive_embodied_ai.representation.movement_features import (
    MovementFeatureExtractor,
)

import cv2



def main():

    camera = Camera()

    tracker = PoseTracker()

    extractor = MovementFeatureExtractor()


    print("Camera started")


    while True:

        frame = camera.read()

        if frame is None:
            break


        result = tracker.detect(frame)


        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]


            features = extractor.extract(
                landmarks
            )


            print(features)



        cv2.imshow(
            "Feature Test",
            frame,
        )


        if cv2.waitKey(1) == 27:
            break


    camera.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()