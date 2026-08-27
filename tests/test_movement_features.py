import pytest

from adaptive_embodied_ai.representation.movement_features import (
    MovementFeatureExtractor,
)


class FakeLandmark:
    """
    Minimal stand-in for a MediaPipe landmark.

    The real MediaPipe landmarks expose x, y, z and visibility,
    which are the properties used by MovementFeatureExtractor.
    """

    def __init__(
        self,
        x,
        y,
        z=0.0,
        visibility=1.0,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def make_landmarks(
    nose,
    left_shoulder,
    right_shoulder,
):
    """
    Build a 33-entry landmark list.

    Only indices 0, 11 and 12 are used by the current extractor.
    """

    landmarks = [
        FakeLandmark(0.5, 0.5, 0.0)
        for _ in range(33)
    ]

    landmarks[0] = nose
    landmarks[11] = left_shoulder
    landmarks[12] = right_shoulder

    return landmarks


def test_first_frame_has_zero_motion():
    extractor = MovementFeatureExtractor()

    landmarks = make_landmarks(
        nose=FakeLandmark(
            0.50,
            0.30,
            -0.10,
        ),
        left_shoulder=FakeLandmark(
            0.40,
            0.50,
            0.00,
        ),
        right_shoulder=FakeLandmark(
            0.60,
            0.50,
            0.00,
        ),
    )

    features = extractor.extract(
        landmarks,
        timestamp=0.0,
    )

    assert features is not None

    assert features["torso_dx"] == 0.0
    assert features["torso_dy"] == 0.0
    assert features["torso_dz"] == 0.0

    assert features["movement_speed"] == 0.0
    assert features["movement_speed_3d"] == 0.0


def test_shoulder_width_and_head_offset_are_normalized_correctly():
    extractor = MovementFeatureExtractor()

    left_shoulder = FakeLandmark(
        0.40,
        0.50,
        0.00,
    )

    right_shoulder = FakeLandmark(
        0.60,
        0.50,
        0.00,
    )

    nose = FakeLandmark(
        0.55,
        0.30,
        -0.10,
    )

    landmarks = make_landmarks(
        nose,
        left_shoulder,
        right_shoulder,
    )

    features = extractor.extract(
        landmarks,
        timestamp=0.0,
    )

    assert features is not None

    expected_shoulder_width = 0.2

    expected_head_offset_x = (
        0.55 - 0.50
    ) / expected_shoulder_width

    expected_head_offset_y = (
        0.30 - 0.50
    ) / expected_shoulder_width

    expected_head_offset_z = (
        -0.10 - 0.00
    ) / expected_shoulder_width

    assert features["shoulder_width"] == pytest.approx(
        expected_shoulder_width
    )

    assert features["head_offset_x"] == pytest.approx(
        expected_head_offset_x
    )

    assert features["head_offset_y"] == pytest.approx(
        expected_head_offset_y
    )

    assert features["head_offset_z"] == pytest.approx(
        expected_head_offset_z
    )


def test_depth_offset_uses_relative_z():
    extractor = MovementFeatureExtractor()

    landmarks = make_landmarks(
        nose=FakeLandmark(
            0.50,
            0.30,
            -0.20,
        ),
        left_shoulder=FakeLandmark(
            0.40,
            0.50,
            0.00,
        ),
        right_shoulder=FakeLandmark(
            0.60,
            0.50,
            0.00,
        ),
    )

    features = extractor.extract(
        landmarks,
        timestamp=0.0,
    )

    assert features is not None

    expected = -0.20 / 0.20

    assert features["head_offset_z"] == pytest.approx(
        expected
    )


def test_temporal_depth_displacement_is_recorded():
    extractor = MovementFeatureExtractor()

    first_frame = make_landmarks(
        nose=FakeLandmark(
            0.50,
            0.30,
            -0.10,
        ),
        left_shoulder=FakeLandmark(
            0.40,
            0.50,
            0.00,
        ),
        right_shoulder=FakeLandmark(
            0.60,
            0.50,
            0.00,
        ),
    )

    second_frame = make_landmarks(
        nose=FakeLandmark(
            0.50,
            0.30,
            -0.10,
        ),
        left_shoulder=FakeLandmark(
            0.40,
            0.50,
            -0.02,
        ),
        right_shoulder=FakeLandmark(
            0.60,
            0.50,
            -0.02,
        ),
    )

    extractor.extract(
        first_frame,
        timestamp=0.0,
    )

    features = extractor.extract(
        second_frame,
        timestamp=1.0,
    )

    assert features is not None

    expected_torso_dz = -0.02 / 0.20

    assert features["torso_dz"] == pytest.approx(
        expected_torso_dz
    )

    assert features["movement_speed_3d"] > 0.0


def test_movement_speed_uses_elapsed_time():
    left_shoulder_1 = FakeLandmark(
        0.40,
        0.50,
        0.00,
    )

    right_shoulder_1 = FakeLandmark(
        0.60,
        0.50,
        0.00,
    )

    nose = FakeLandmark(
        0.50,
        0.30,
        -0.10,
    )

    first_frame = make_landmarks(
        nose,
        left_shoulder_1,
        right_shoulder_1,
    )

    second_frame = make_landmarks(
        nose,
        FakeLandmark(
            0.42,
            0.50,
            0.00,
        ),
        FakeLandmark(
            0.62,
            0.50,
            0.00,
        ),
    )

    extractor_fast = MovementFeatureExtractor()

    extractor_fast.extract(
        first_frame,
        timestamp=0.0,
    )

    features_fast = extractor_fast.extract(
        second_frame,
        timestamp=0.1,
    )

    extractor_slow = MovementFeatureExtractor()

    extractor_slow.extract(
        first_frame,
        timestamp=0.0,
    )

    features_slow = extractor_slow.extract(
        second_frame,
        timestamp=1.0,
    )

    assert features_fast is not None
    assert features_slow is not None

    displacement = 0.02 / 0.2

    assert features_fast["movement_speed"] == pytest.approx(
        displacement / 0.1
    )

    assert features_slow["movement_speed"] == pytest.approx(
        displacement / 1.0
    )

    assert features_fast["movement_speed"] > (
        features_slow["movement_speed"]
    )


def test_3d_speed_is_at_least_2d_speed_when_depth_changes():
    first_frame = make_landmarks(
        nose=FakeLandmark(
            0.50,
            0.30,
            -0.10,
        ),
        left_shoulder=FakeLandmark(
            0.40,
            0.50,
            0.00,
        ),
        right_shoulder=FakeLandmark(
            0.60,
            0.50,
            0.00,
        ),
    )

    second_frame = make_landmarks(
        nose=FakeLandmark(
            0.50,
            0.30,
            -0.10,
        ),
        left_shoulder=FakeLandmark(
            0.42,
            0.50,
            -0.02,
        ),
        right_shoulder=FakeLandmark(
            0.62,
            0.50,
            -0.02,
        ),
    )

    extractor = MovementFeatureExtractor()

    extractor.extract(
        first_frame,
        timestamp=0.0,
    )

    features = extractor.extract(
        second_frame,
        timestamp=1.0,
    )

    assert features is not None

    assert (
        features["movement_speed_3d"]
        >= features["movement_speed"]
    )


def test_low_visibility_shoulder_returns_none():
    extractor = MovementFeatureExtractor()

    landmarks = make_landmarks(
        nose=FakeLandmark(
            0.50,
            0.30,
            -0.10,
        ),
        left_shoulder=FakeLandmark(
            0.40,
            0.50,
            0.00,
            visibility=0.4,
        ),
        right_shoulder=FakeLandmark(
            0.60,
            0.50,
            0.00,
            visibility=1.0,
        ),
    )

    features = extractor.extract(
        landmarks,
        timestamp=0.0,
    )

    assert features is None


def test_reset_clears_temporal_state():
    extractor = MovementFeatureExtractor()

    frame_1 = make_landmarks(
        nose=FakeLandmark(
            0.50,
            0.30,
            -0.10,
        ),
        left_shoulder=FakeLandmark(
            0.40,
            0.50,
            0.00,
        ),
        right_shoulder=FakeLandmark(
            0.60,
            0.50,
            0.00,
        ),
    )

    frame_2 = make_landmarks(
        nose=FakeLandmark(
            0.50,
            0.30,
            -0.10,
        ),
        left_shoulder=FakeLandmark(
            0.42,
            0.50,
            -0.02,
        ),
        right_shoulder=FakeLandmark(
            0.62,
            0.50,
            -0.02,
        ),
    )

    extractor.extract(
        frame_1,
        timestamp=0.0,
    )

    extractor.extract(
        frame_2,
        timestamp=1.0,
    )

    extractor.reset()

    features = extractor.extract(
        frame_1,
        timestamp=0.0,
    )

    assert features is not None

    assert features["torso_dx"] == 0.0
    assert features["torso_dy"] == 0.0
    assert features["torso_dz"] == 0.0
    assert features["movement_speed"] == 0.0
    assert features["movement_speed_3d"] == 0.0