import math


class MovementFeatureExtractor:
    """
    Convert MediaPipe pose landmarks into normalized movement features.

    The extractor keeps the original 2D movement representation and adds
    relative-depth (z) information from MediaPipe.

    Important:
    MediaPipe's z coordinate is a relative depth estimate. It is not
    metric 3D reconstruction. Therefore the features produced here are
    described as depth-aware / relative-depth features.

    Spatial features are normalized by shoulder width to reduce sensitivity
    to camera distance and differences in body size.
    """

    VISIBILITY_THRESHOLD = 0.5
    EPSILON = 1e-6

    def __init__(self):
        self.previous_center = None
        self.previous_timestamp = None

    def reset(self):
        """Reset temporal state before starting a new recording."""
        self.previous_center = None
        self.previous_timestamp = None

    def extract(self, landmarks, timestamp):
        """
        Extract movement features from one pose observation.

        Parameters
        ----------
        landmarks : list
            List of 33 MediaPipe pose landmarks.

        timestamp : float
            Seconds elapsed since the beginning of the recording.

        Returns
        -------
        dict or None
            Feature dictionary, or None when the required landmarks are
            not sufficiently visible.
        """

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        nose = landmarks[0]

        if (
            left_shoulder.visibility < self.VISIBILITY_THRESHOLD
            or right_shoulder.visibility < self.VISIBILITY_THRESHOLD
        ):
            return None

    
        center_x = (left_shoulder.x + right_shoulder.x) / 2
        center_y = (left_shoulder.y + right_shoulder.y) / 2
        center_z = (left_shoulder.z + right_shoulder.z) / 2

        current_center = (center_x, center_y, center_z)

        shoulder_width = self.distance_2d(
            left_shoulder,
            right_shoulder,
        )

        if shoulder_width < self.EPSILON:
            return None

        head_offset_x = (
            nose.x - center_x
        ) / shoulder_width

        head_offset_y = (
            nose.y - center_y
        ) / shoulder_width

        head_offset_z = (
            nose.z - center_z
        ) / shoulder_width

        if (
            self.previous_center is not None
            and self.previous_timestamp is not None
        ):
            dt = timestamp - self.previous_timestamp

            torso_dx = (
                current_center[0] - self.previous_center[0]
            ) / shoulder_width

            torso_dy = (
                current_center[1] - self.previous_center[1]
            ) / shoulder_width

            torso_dz = (
                current_center[2] - self.previous_center[2]
            ) / shoulder_width

            displacement_2d = math.sqrt(
                torso_dx**2 + torso_dy**2
            )

            displacement_3d = math.sqrt(
                torso_dx**2
                + torso_dy**2
                + torso_dz**2
            )

            if dt > self.EPSILON:
                movement_speed_2d = displacement_2d / dt
                movement_speed_3d = displacement_3d / dt
            else:
                movement_speed_2d = 0.0
                movement_speed_3d = 0.0

        else:
            torso_dx = 0.0
            torso_dy = 0.0
            torso_dz = 0.0
            movement_speed_2d = 0.0
            movement_speed_3d = 0.0

        features = {
            # 2D representation
            "shoulder_width": shoulder_width,
            "head_offset_x": head_offset_x,
            "head_offset_y": head_offset_y,
            "torso_dx": torso_dx,
            "torso_dy": torso_dy,
            "movement_speed": movement_speed_2d,

            # depth-aware representation
            "head_offset_z": head_offset_z,
            "torso_dz": torso_dz,
            "movement_speed_3d": movement_speed_3d,
        }

        self.previous_center = current_center
        self.previous_timestamp = timestamp

        return features

    @staticmethod
    def distance_2d(point_a, point_b):
        """Euclidean distance using x/y coordinates."""
        return math.sqrt(
            (point_a.x - point_b.x) ** 2
            + (point_a.y - point_b.y) ** 2
        )

    @staticmethod
    def distance_3d(point_a, point_b):
        """Euclidean distance using x/y/z coordinates."""
        return math.sqrt(
            (point_a.x - point_b.x) ** 2
            + (point_a.y - point_b.y) ** 2
            + (point_a.z - point_b.z) ** 2
        )