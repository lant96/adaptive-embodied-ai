import math

import pytest

from adaptive_embodied_ai.representation.movement_features import (
    MovementFeatureExtractor,
)


class FakeLandmark:
    """
    Minimal stand-in for a MediaPipe landmark.
    Only x, y, and visibility are used by MovementFeatureExtractor.
    """

    def __init__(self, x, y, visibility=1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def make_landmarks(nose, left_shoulder, right_shoulder):
    """
    Builds a 33-entry landmark list with the three landmarks
    MovementFeatureExtractor actually reads (indices 0, 11, 12)
    set explicitly, and harmless placeholders everywhere else.
    """
    landmarks = [FakeLandmark(0.5, 0.5) for _ in range(33)]
    landmarks[0] = nose
    landmarks[11] = left_shoulder
    landmarks[12] = right_shoulder
    return landmarks


def test_first_frame_has_zero_motion():
    """
    With no previous frame to compare against, torso_dx/dy and
    movement_speed must be zero rather than crashing or guessing.
    """
    extractor = MovementFeatureExtractor()

    landmarks = make_landmarks(
        nose=FakeLandmark(0.50, 0.30),
        left_shoulder=FakeLandmark(0.40, 0.50),
        right_shoulder=FakeLandmark(0.60, 0.50),
    )

    features = extractor.extract(landmarks, timestamp=0.0)

    assert features is not None
    assert features["torso_dx"] == 0.0
    assert features["torso_dy"] == 0.0
    assert features["movement_speed"] == 0.0


def test_shoulder_width_and_head_offset_are_normalized_correctly():
    extractor = MovementFeatureExtractor()

    left_shoulder = FakeLandmark(0.40, 0.50)
    right_shoulder = FakeLandmark(0.60, 0.50)
    nose = FakeLandmark(0.55, 0.30)

    landmarks = make_landmarks(nose, left_shoulder, right_shoulder)
    features = extractor.extract(landmarks, timestamp=0.0)
    assert features is not None

    expected_shoulder_width = 0.2
    expected_head_offset_x = (0.55 - 0.50) / expected_shoulder_width
    expected_head_offset_y = (0.30 - 0.50) / expected_shoulder_width

    assert features["shoulder_width"] == pytest.approx(expected_shoulder_width)
    assert features["head_offset_x"] == pytest.approx(expected_head_offset_x)
    assert features["head_offset_y"] == pytest.approx(expected_head_offset_y)


def test_movement_speed_uses_elapsed_time_not_just_displacement():
    left_shoulder = FakeLandmark(0.40, 0.50)
    right_shoulder = FakeLandmark(0.60, 0.50)
    nose = FakeLandmark(0.50, 0.30)
    shoulder_width = 0.2

    landmarks_frame_1 = make_landmarks(nose, left_shoulder, right_shoulder)
    landmarks_frame_2 = make_landmarks(
        nose,
        FakeLandmark(0.42, 0.50),
        FakeLandmark(0.62, 0.50),
    )

    extractor_fast = MovementFeatureExtractor()
    extractor_fast.extract(landmarks_frame_1, timestamp=0.0)
    features_fast = extractor_fast.extract(landmarks_frame_2, timestamp=0.1)
    assert features_fast is not None

    extractor_slow = MovementFeatureExtractor()
    extractor_slow.extract(landmarks_frame_1, timestamp=0.0)
    features_slow = extractor_slow.extract(landmarks_frame_2, timestamp=1.0)
    assert features_slow is not None

    displacement_in_body_units = 0.02 / shoulder_width

    assert features_fast["movement_speed"] == pytest.approx(
        displacement_in_body_units / 0.1
    )
    assert features_slow["movement_speed"] == pytest.approx(
        displacement_in_body_units / 1.0
    )
    assert features_fast["movement_speed"] > features_slow["movement_speed"]


def test_low_visibility_shoulder_returns_none():
    """
    A shoulder MediaPipe isn't confident about shouldn't silently produce
    a feature row - extract() should signal "skip this frame" via None.
    """
    extractor = MovementFeatureExtractor()

    landmarks = make_landmarks(
        nose=FakeLandmark(0.50, 0.30),
        left_shoulder=FakeLandmark(0.40, 0.50, visibility=0.1),
        right_shoulder=FakeLandmark(0.60, 0.50, visibility=0.9),
    )

    features = extractor.extract(landmarks, timestamp=0.0)

    assert features is None


def test_degenerate_zero_width_shoulders_returns_none():
    """
    If both shoulder landmarks collapse to the same point (e.g. a bad
    detection), shoulder_width is ~0 and normalizing by it would divide
    by zero - extract() must guard against this instead of crashing.
    """
    extractor = MovementFeatureExtractor()

    same_point = FakeLandmark(0.50, 0.50)
    landmarks = make_landmarks(
        nose=FakeLandmark(0.50, 0.30),
        left_shoulder=same_point,
        right_shoulder=same_point,
    )

    features = extractor.extract(landmarks, timestamp=0.0)

    assert features is None


def test_reset_clears_previous_state():
    extractor = MovementFeatureExtractor()

    landmarks = make_landmarks(
        nose=FakeLandmark(0.50, 0.30),
        left_shoulder=FakeLandmark(0.40, 0.50),
        right_shoulder=FakeLandmark(0.60, 0.50),
    )

    extractor.extract(landmarks, timestamp=0.0)
    extractor.extract(landmarks, timestamp=0.1)

    extractor.reset()
    features_after_reset = extractor.extract(landmarks, timestamp=5.0)
    assert features_after_reset is not None

    assert features_after_reset["torso_dx"] == 0.0
    assert features_after_reset["torso_dy"] == 0.0
    assert features_after_reset["movement_speed"] == 0.0