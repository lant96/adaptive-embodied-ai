from sklearn.preprocessing import StandardScaler


class MovementNormalizer:
    """
    Standardizes movement features using statistics
    learned from training data only.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, features):
        """
        Learn normalization parameters from training data.
        """
        self.scaler.fit(features)
        self.fitted = True

        return self

    def transform(self, features):
        """
        Transform features using the parameters learned
        during fitting.
        """
        if not self.fitted:
            raise RuntimeError(
                "MovementNormalizer must be fitted before transform()."
            )

        return self.scaler.transform(features)

    def fit_transform(self, features):
        """
        Fit the normalizer and transform the same data.
        Intended for training data.
        """
        self.fit(features)

        return self.transform(features)