from pathlib import Path

import cv2
import mediapipe as mp


class PoseTracker:

    def __init__(self):
        from adaptive_embodied_ai.utils.paths import MODELS_DIR
        model_path = MODELS_DIR / "pose_landmarker_lite.task"

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
        )

        self.detector = PoseLandmarker.create_from_options(options)

    def detect(self, frame, timestamp_ms):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect_for_video(image, timestamp_ms)