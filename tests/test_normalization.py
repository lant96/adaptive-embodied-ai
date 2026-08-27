import numpy as np

from adaptive_embodied_ai.representation.normalization import (
    MovementNormalizer,
)


def test_normalizer_fit_transform():

    features = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    normalizer = MovementNormalizer()

    transformed = normalizer.fit_transform(features)

    assert transformed.shape == features.shape

    assert np.allclose(
        transformed.mean(axis=0),
        0.0,
    )

    assert np.allclose(
        transformed.std(axis=0),
        1.0,
    )


def test_transform_requires_fit():

    normalizer = MovementNormalizer()

    features = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    try:
        normalizer.transform(features)
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass