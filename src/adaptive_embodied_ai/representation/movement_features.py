import math


class MovementFeatureExtractor:
    """
    Converts MediaPipe pose landmarks
    into human-centered movement features.
    """

    def __init__(self):
        self.previous_center = None


    def extract(self, landmarks):
        """
        Extract movement representation
        from MediaPipe landmarks.

        landmarks:
            list of 33 MediaPipe landmarks
        """

        features = {}

        # -----------------------------
        # Shoulder center
        # -----------------------------

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        center_x = (
            left_shoulder.x +
            right_shoulder.x
        ) / 2

        center_y = (
            left_shoulder.y +
            right_shoulder.y
        ) / 2


        features["torso_x"] = center_x
        features["torso_y"] = center_y


        # -----------------------------
        # Shoulder width
        # -----------------------------

        shoulder_distance = self.distance(
            left_shoulder,
            right_shoulder
        )

        features["shoulder_width"] = shoulder_distance


        # -----------------------------
        # Head movement proxy
        # -----------------------------

        nose = landmarks[0]

        head_offset_x = nose.x - center_x
        head_offset_y = nose.y - center_y


        features["head_offset_x"] = head_offset_x
        features["head_offset_y"] = head_offset_y


        # -----------------------------
        # Movement velocity
        # -----------------------------

        current_position = (
            center_x,
            center_y
        )


        if self.previous_center:

            velocity = self.distance_points(
                current_position,
                self.previous_center
            )

        else:
            velocity = 0


        features["movement_speed"] = velocity


        self.previous_center = current_position


        return features



    def distance(self, point_a, point_b):
        """
        Distance between MediaPipe landmarks.
        """

        return math.sqrt(
            (point_a.x - point_b.x)**2 +
            (point_a.y - point_b.y)**2
        )


    def distance_points(self, a, b):

        return math.sqrt(
            (a[0]-b[0])**2 +
            (a[1]-b[1])**2
        )