import time

from adaptive_embodied_ai.acquisition.camera import Camera
from adaptive_embodied_ai.acquisition.pose_tracker import PoseTracker

from adaptive_embodied_ai.representation.movement_features import (
    MovementFeatureExtractor,
)

from adaptive_embodied_ai.acquisition.feature_recorder import (
    FeatureRecorder,
)

import cv2



def main():

    camera = Camera()

    tracker = PoseTracker()

    extractor = MovementFeatureExtractor()

    recorder = FeatureRecorder(
        "data/movement/session_01.csv"
    )


    print(
        "Recording started. Press ESC to stop."
    )


    start_time = time.time()


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


            timestamp = (
                time.time() - start_time
            )


            recorder.record(
                timestamp,
                features
            )


            print(features)



        cv2.imshow(
            "Movement Recorder",
            frame
        )


        if cv2.waitKey(1) == 27:
            break



    recorder.close()

    camera.release()

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()