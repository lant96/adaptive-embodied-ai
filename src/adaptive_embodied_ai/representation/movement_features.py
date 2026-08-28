import math


class MovementFeatureExtractor:
    """
    Convert MediaPipe pose landmarks into normalized movement features.

    Two coordinate spaces are used deliberately:

    1. Image-space landmarks
       Used for the 2D representation:
       - shoulder width
       - head offset x/y
       - torso displacement x/y
       - 2D movement speed

    2. World-space landmarks
       Used for the depth-aware / 3D representation:
       - head offset z
       - torso displacement z
       - 3D movement speed

    MediaPipe provides pose_world_landmarks as real-world 3D
    coordinates in meters, with the origin at the midpoint
    between the hips.

    Image-space z is therefore NOT used as the source of the
    project's depth representation.

    Spatial features are normalized by shoulder width in the
    corresponding coordinate space to reduce sensitivity to
    body size and camera distance.
    """

    VISIBILITY_THRESHOLD = 0.5
    EPSILON = 1e-6

    def __init__(self):
        self.previous_center = None
        self.previous_world_center = None
        self.previous_timestamp = None

    def reset(self):
        """Reset temporal state before a new recording."""

        self.previous_center = None
        self.previous_world_center = None
        self.previous_timestamp = None

    def extract(
        self,
        landmarks,
        world_landmarks,
        timestamp,
    ):
        """
        Extract movement features from one pose observation.

        Parameters
        ----------
        landmarks : list
            MediaPipe normalized image-space landmarks.

        world_landmarks : list
            MediaPipe world-space landmarks.

        timestamp : float
            Seconds elapsed since the beginning of the recording.

        Returns
        -------
        dict or None
            Extracted feature dictionary, or None if required
            landmarks are not sufficiently visible.
        """

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        nose = landmarks[0]

        world_left_shoulder = world_landmarks[11]
        world_right_shoulder = world_landmarks[12]
        world_nose = world_landmarks[0]

        if (
            left_shoulder.visibility < self.VISIBILITY_THRESHOLD
            or right_shoulder.visibility < self.VISIBILITY_THRESHOLD
        ):
            return None

        if (
            world_left_shoulder.visibility
            < self.VISIBILITY_THRESHOLD
            or world_right_shoulder.visibility
            < self.VISIBILITY_THRESHOLD
        ):
            return None

        # Image-space representation
        
        center_x = (
            left_shoulder.x + right_shoulder.x
        ) / 2

        center_y = (
            left_shoulder.y + right_shoulder.y
        ) / 2

        current_center = (
            center_x,
            center_y,
        )

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

        # World-space 3D representation
        
        world_center = (
            (
                world_left_shoulder.x
                + world_right_shoulder.x
            ) / 2,

            (
                world_left_shoulder.y
                + world_right_shoulder.y
            ) / 2,

            (
                world_left_shoulder.z
                + world_right_shoulder.z
            ) / 2,
        )

        world_shoulder_width = self.distance_3d(
            world_left_shoulder,
            world_right_shoulder,
        )

        if world_shoulder_width < self.EPSILON:
            return None

        head_offset_z = (
            world_nose.z - world_center[2]
        ) / world_shoulder_width

        if (
            self.previous_center is not None
            and self.previous_world_center is not None
            and self.previous_timestamp is not None
        ):
            dt = (
                timestamp
                - self.previous_timestamp
            )

            # 2D displacement
            torso_dx = (
                current_center[0]
                - self.previous_center[0]
            ) / shoulder_width

            torso_dy = (
                current_center[1]
                - self.previous_center[1]
            ) / shoulder_width

            displacement_2d = math.sqrt(
                torso_dx**2
                + torso_dy**2
            )

            # 3D displacement
            world_dx = (
                world_center[0]
                - self.previous_world_center[0]
            ) / world_shoulder_width

            world_dy = (
                world_center[1]
                - self.previous_world_center[1]
            ) / world_shoulder_width

            torso_dz = (
                world_center[2]
                - self.previous_world_center[2]
            ) / world_shoulder_width

            displacement_3d = math.sqrt(
                world_dx**2
                + world_dy**2
                + torso_dz**2
            )

            if dt > self.EPSILON:
                movement_speed = (
                    displacement_2d / dt
                )

                movement_speed_3d = (
                    displacement_3d / dt
                )
            else:
                movement_speed = 0.0
                movement_speed_3d = 0.0

        else:
            torso_dx = 0.0
            torso_dy = 0.0
            torso_dz = 0.0

            movement_speed = 0.0
            movement_speed_3d = 0.0

        features = {
            "shoulder_width": shoulder_width,
            "head_offset_x": head_offset_x,
            "head_offset_y": head_offset_y,
            "torso_dx": torso_dx,
            "torso_dy": torso_dy,
            "movement_speed": movement_speed,
            "head_offset_z": head_offset_z,
            "torso_dz": torso_dz,
            "movement_speed_3d": movement_speed_3d,
        }

        self.previous_center = current_center
        self.previous_world_center = world_center
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