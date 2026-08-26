import math


class MovementFeatureExtractor:
    """
    Converts MediaPipe pose landmarks into
    movement features for adaptive embodied interaction.

    Distances are expressed in "shoulder widths" rather than
    raw frame-fraction units, so features are roughly invariant
    to camera distance and body size.
    """

    VISIBILITY_THRESHOLD = 0.5

    def __init__(self):
        self.previous_center = None
        self.previous_timestamp = None

    def reset(self):
        """Call this at the start of every new recording session."""
        self.previous_center = None
        self.previous_timestamp = None

    def extract(self, landmarks, timestamp):
        """
        Parameters
        ----------
        landmarks : list of 33 MediaPipe pose landmarks.
        timestamp : float, seconds elapsed since session start.

        Returns
        -------
        dict, or None if the required landmarks aren't confidently visible.
        """
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        nose = landmarks[0]

        if (left_shoulder.visibility < self.VISIBILITY_THRESHOLD or
                right_shoulder.visibility < self.VISIBILITY_THRESHOLD):
            return None

        center_x = (left_shoulder.x + right_shoulder.x) / 2
        center_y = (left_shoulder.y + right_shoulder.y) / 2
        current_center = (center_x, center_y)

        shoulder_width = self.distance(left_shoulder, right_shoulder)
        if shoulder_width < 1e-6:
            return None

        features = {
            "shoulder_width": shoulder_width,
            "head_offset_x": (nose.x - center_x) / shoulder_width,
            "head_offset_y": (nose.y - center_y) / shoulder_width,
        }

        if self.previous_center is not None and self.previous_timestamp is not None:
            dt = timestamp - self.previous_timestamp
            torso_dx = (current_center[0] - self.previous_center[0]) / shoulder_width
            torso_dy = (current_center[1] - self.previous_center[1]) / shoulder_width
            displacement = math.sqrt(torso_dx**2 + torso_dy**2)
            # dt guard: avoids inflated speed after a dropped-frame gap
            movement_speed = displacement / dt if dt > 1e-6 else 0.0
        else:
            torso_dx = torso_dy = movement_speed = 0.0

        features["torso_dx"] = torso_dx
        features["torso_dy"] = torso_dy
        features["movement_speed"] = movement_speed

        self.previous_center = current_center
        self.previous_timestamp = timestamp

        return features

    @staticmethod
    def distance(point_a, point_b):
        return math.sqrt(
            (point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2
        )