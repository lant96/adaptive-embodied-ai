from pathlib import Path

import cv2
import mediapipe as mp


class PoseTracker:
    """
    MediaPipe Pose Landmarker wrapper.

    The tracker uses the Full Pose Landmarker model and exposes both:
    - normalized image-space pose landmarks
    - world-space pose landmarks

    World landmarks are used downstream for 3D movement features.
    """

    def __init__(self):
        from adaptive_embodied_ai.utils.paths import MODELS_DIR

        model_path = MODELS_DIR / "pose_landmarker_full.task"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found: {model_path}"
            )

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
        )

        self.detector = PoseLandmarker.create_from_options(
            options
        )

    def detect(self, frame, timestamp_ms):
        """
        Detect pose landmarks for one video frame.

        Returns a MediaPipe PoseLandmarkerResult containing:
        - pose_landmarks
        - pose_world_landmarks
        """

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        return self.detector.detect_for_video(
            image,
            timestamp_ms,
        )