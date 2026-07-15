"""
MediaPipe pose acquisition module.

Responsible for:
- Loading the MediaPipe PoseLandmarker model
- Processing camera frames
- Returning human pose landmarks

This module does NOT perform:
- gesture recognition
- personalization
- interaction logic
"""

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp


class PoseTracker:
    """
    Wrapper around MediaPipe PoseLandmarker.

    Input:
        RGB image frame

    Output:
        Pose landmarks (33 body keypoints)
    """

    def __init__(self):
        self.pose_landmarker = self._initialize_model()

    def _initialize_model(self):
        """
        Initialize MediaPipe PoseLandmarker.
        """

        base_options = mp.tasks.BaseOptions(
            model_asset_path=self._download_model()
        )

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
        )

        detector = mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        )

        return detector

    def _download_model(self) -> str:
        """
        Return path to MediaPipe pose model.

        For now we keep the model locally.
        Later we can improve model management.
        """

        model_path = Path("models/pose_landmarker_lite.task")

        if not model_path.exists():
            raise FileNotFoundError(
                """
                MediaPipe model not found.

                Download:
                pose_landmarker_lite.task

                and place it in:

                models/pose_landmarker_lite.task
                """
            )

        return str(model_path)

    def process(self, frame):
        """
        Process a camera frame.

        Args:
            frame:
                OpenCV BGR image

        Returns:
            MediaPipe detection result
        """

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = int(
            cv2.getTickCount()
            /
            cv2.getTickFrequency()
            *
            1000
        )

        result = self.pose_landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        return result

    def extract_landmarks(self, result):
        """
        Convert MediaPipe output into
        a simple list of landmarks.

        Returns:
            List of dictionaries
        """

        if not result.pose_landmarks:
            return None

        landmarks = []

        for landmark in result.pose_landmarks[0]:
            landmarks.append(
                {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                }
            )

        return landmarks