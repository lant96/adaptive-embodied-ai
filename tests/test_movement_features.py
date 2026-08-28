from dataclasses import dataclass

import pytest

from adaptive_embodied_ai.representation.movement_features import (
    MovementFeatureExtractor,
)


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float
    visibility: float = 1.0


def make_landmarks(
    nose=None,
    left_shoulder=None,
    right_shoulder=None,
):
    landmarks = [
        FakeLandmark(0.5, 0.5, 0.0)
        for _ in range(33)
    ]

    landmarks[0] = nose or FakeLandmark(0.5, 0.3, -0.1)
    landmarks[11] = left_shoulder or FakeLandmark(0.4, 0.5, 0.0)
    landmarks[12] = right_shoulder or FakeLandmark(0.6, 0.5, 0.0)

    return landmarks


def make_world_landmarks(
    nose=None,
    left_shoulder=None,
    right_shoulder=None,
):
    landmarks = [
        FakeLandmark(0.0, 0.0, 0.0)
        for _ in range(33)
    ]

    landmarks[0] = nose or FakeLandmark(0.0, -0.4, -0.1)
    landmarks[11] = left_shoulder or FakeLandmark(-0.2, 0.0, 0.0)
    landmarks[12] = right_shoulder or FakeLandmark(0.2, 0.0, 0.0)

    return landmarks


def test_first_frame_has_zero_motion():
    extractor = MovementFeatureExtractor()

    landmarks = make_landmarks()
    world_landmarks = make_world_landmarks()

    features = extractor.extract(
        landmarks,
        world_landmarks,
        timestamp=0.0,
    )

    assert features is not None

    assert features["torso_dx"] == pytest.approx(0.0)
    assert features["torso_dy"] == pytest.approx(0.0)
    assert features["torso_dz"] == pytest.approx(0.0)

    assert features["movement_speed"] == pytest.approx(0.0)
    assert features["movement_speed_3d"] == pytest.approx(0.0)


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

    world_landmarks = make_world_landmarks()

    features = extractor.extract(
        landmarks,
        world_landmarks,
        timestamp=0.0,
    )

    assert features is not None

    assert features["shoulder_width"] == pytest.approx(0.20)

    assert features["head_offset_x"] == pytest.approx(0.25)
    assert features["head_offset_y"] == pytest.approx(-1.0)


def test_depth_offset_uses_world_coordinates():
    extractor = MovementFeatureExtractor()

    landmarks = make_landmarks()

    world_landmarks = make_world_landmarks(
        nose=FakeLandmark(
            0.0,
            -0.4,
            -0.20,
        ),
        left_shoulder=FakeLandmark(
            -0.20,
            0.0,
            0.0,
        ),
        right_shoulder=FakeLandmark(
            0.20,
            0.0,
            0.0,
        ),
    )

    features = extractor.extract(
        landmarks,
        world_landmarks,
        timestamp=0.0,
    )

    assert features is not None

    # World-space shoulder width = 0.40 m.
    # Nose is 0.20 m in front of the shoulder midpoint.
    expected_depth_offset = -0.20 / 0.40

    assert features["head_offset_z"] == pytest.approx(
        expected_depth_offset
    )


def test_temporal_depth_displacement_is_recorded():
    extractor = MovementFeatureExtractor()

    landmarks = make_landmarks()

    first_world_frame = make_world_landmarks(
        left_shoulder=FakeLandmark(
            -0.20,
            0.0,
            0.0,
        ),
        right_shoulder=FakeLandmark(
            0.20,
            0.0,
            0.0,
        ),
    )

    second_world_frame = make_world_landmarks(
        left_shoulder=FakeLandmark(
            -0.20,
            0.0,
            -0.02,
        ),
        right_shoulder=FakeLandmark(
            0.20,
            0.0,
            -0.02,
        ),
    )

    extractor.extract(
        landmarks,
        first_world_frame,
        timestamp=0.0,
    )

    features = extractor.extract(
        landmarks,
        second_world_frame,
        timestamp=1.0,
    )

    assert features is not None

    # Both shoulders move 0.02 m in depth.
    # Shoulder width = 0.40 m.
    expected_dz = -0.02 / 0.40

    assert features["torso_dz"] == pytest.approx(
        expected_dz
    )


def test_movement_speed_uses_elapsed_time():
    first_landmarks = make_landmarks(
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

    second_landmarks = make_landmarks(
        left_shoulder=FakeLandmark(
            0.42,
            0.50,
            0.00,
        ),
        right_shoulder=FakeLandmark(
            0.62,
            0.50,
            0.00,
        ),
    )

    world_landmarks = make_world_landmarks()

    extractor_fast = MovementFeatureExtractor()

    extractor_fast.extract(
        first_landmarks,
        world_landmarks,
        timestamp=0.0,
    )

    fast_features = extractor_fast.extract(
        second_landmarks,
        world_landmarks,
        timestamp=1.0,
    )

    extractor_slow = MovementFeatureExtractor()

    extractor_slow.extract(
        first_landmarks,
        world_landmarks,
        timestamp=0.0,
    )

    slow_features = extractor_slow.extract(
        second_landmarks,
        world_landmarks,
        timestamp=2.0,
    )

    assert fast_features is not None
    assert slow_features is not None

    assert fast_features["movement_speed"] == pytest.approx(
        slow_features["movement_speed"] * 2
    )


def test_3d_speed_is_at_least_2d_speed_when_depth_changes():
    first_landmarks = make_landmarks(
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

    second_landmarks = make_landmarks(
        left_shoulder=FakeLandmark(
            0.42,
            0.50,
            0.00,
        ),
        right_shoulder=FakeLandmark(
            0.62,
            0.50,
            0.00,
        ),
    )

    first_world = make_world_landmarks(
        left_shoulder=FakeLandmark(
            -0.20,
            0.0,
            0.0,
        ),
        right_shoulder=FakeLandmark(
            0.20,
            0.0,
            0.0,
        ),
    )

    second_world = make_world_landmarks(
        left_shoulder=FakeLandmark(
            -0.20,
            0.0,
            -0.05,
        ),
        right_shoulder=FakeLandmark(
            0.20,
            0.0,
            -0.05,
        ),
    )

    extractor = MovementFeatureExtractor()

    extractor.extract(
        first_landmarks,
        first_world,
        timestamp=0.0,
    )

    features = extractor.extract(
        second_landmarks,
        second_world,
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

    world_landmarks = make_world_landmarks()

    features = extractor.extract(
        landmarks,
        world_landmarks,
        timestamp=0.0,
    )

    assert features is None


def test_reset_clears_temporal_state():
    extractor = MovementFeatureExtractor()

    frame_1 = make_landmarks()

    frame_2 = make_landmarks(
        left_shoulder=FakeLandmark(
            0.42,
            0.50,
            0.00,
        ),
        right_shoulder=FakeLandmark(
            0.62,
            0.50,
            0.00,
        ),
    )

    world_frame_1 = make_world_landmarks()

    world_frame_2 = make_world_landmarks(
        left_shoulder=FakeLandmark(
            -0.20,
            0.0,
            -0.05,
        ),
        right_shoulder=FakeLandmark(
            0.20,
            0.0,
            -0.05,
        ),
    )

    extractor.extract(
        frame_1,
        world_frame_1,
        timestamp=0.0,
    )

    extractor.extract(
        frame_2,
        world_frame_2,
        timestamp=1.0,
    )

    extractor.reset()

    features = extractor.extract(
        frame_2,
        world_frame_2,
        timestamp=2.0,
    )

    assert features is not None

    assert features["torso_dx"] == pytest.approx(0.0)
    assert features["torso_dy"] == pytest.approx(0.0)
    assert features["torso_dz"] == pytest.approx(0.0)

    assert features["movement_speed"] == pytest.approx(0.0)
    assert features["movement_speed_3d"] == pytest.approx(0.0)