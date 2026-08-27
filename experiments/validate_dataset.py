from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data") / "movement"

DATASET_FILES = [
    "P01_session_01.csv",
    "P01_session_02.csv",
    "P02_session_01.csv",
    "P02_session_02.csv",
]

EXPECTED_MOVEMENTS = [
    "neutral",
    "raise_arms",
    "lean_left",
    "lean_right",
    "lean_forward",
]

IDENTIFIER_COLUMNS = [
    "participant_id",
    "session_id",
    "trial_id",
    "movement_label",
]

FEATURE_COLUMNS = [
    "shoulder_width",
    "head_offset_x",
    "head_offset_y",
    "torso_dx",
    "torso_dy",
    "movement_speed",
    "head_offset_z",
    "torso_dz",
    "movement_speed_3d",
]

REQUIRED_COLUMNS = [
    *IDENTIFIER_COLUMNS,
    "timestamp",
    *FEATURE_COLUMNS,
]


def load_dataset():
    """Load all experimental participant/session files."""

    frames = []

    print("=" * 70)
    print("Loading experimental dataset")
    print("=" * 70)

    for filename in DATASET_FILES:
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset file: {path}"
            )

        df = pd.read_csv(path)

        print(
            f"{filename:<25} "
            f"{len(df):>6} rows"
        )

        frames.append(df)

    dataset = pd.concat(
        frames,
        ignore_index=True,
    )

    return dataset


def check_required_columns(df):
    """Check that all expected columns exist."""

    print()
    print("=" * 70)
    print("COLUMN CHECK")
    print("=" * 70)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        print("FAILED")
        print("Missing columns:")

        for column in missing:
            print(f"  - {column}")

        return False

    print("PASS - all required columns are present.")

    return True

def check_identifiers(df):
    """Check participant, session and trial identifiers."""

    print()
    print("=" * 70)
    print("IDENTIFIER CHECK")
    print("=" * 70)

    participants = df["participant_id"].nunique()

    sessions = df[
        ["participant_id", "session_id"]
    ].drop_duplicates()

    trials = df[
        [
            "participant_id",
            "session_id",
            "trial_id",
        ]
    ].drop_duplicates()

    print(f"Participants: {participants}")
    print(f"Sessions:     {len(sessions)}")
    print(f"Trials:       {len(trials)}")

    print()
    print("Participants:")

    for participant in sorted(
        df["participant_id"].unique()
    ):
        print(f"  - {participant}")

    return True


def check_trial_structure(df):
    """
    Check number of trials and movement classes
    for every participant/session.
    """

    print()
    print("=" * 70)
    print("TRIAL STRUCTURE")
    print("=" * 70)

    trial_table = (
        df[
            [
                "participant_id",
                "session_id",
                "trial_id",
                "movement_label",
            ]
        ]
        .drop_duplicates()
    )

    grouped = (
        trial_table
        .groupby(
            [
                "participant_id",
                "session_id",
            ]
        )
        .agg(
            trials=("trial_id", "nunique"),
            movements=("movement_label", "nunique"),
        )
        .reset_index()
    )

    print(grouped.to_string(index=False))

    print()

    expected_trials = len(EXPECTED_MOVEMENTS) * 5

    passed = True

    for _, row in grouped.iterrows():

        if row["trials"] != expected_trials:
            print(
                f"WARNING: "
                f"{row['participant_id']} "
                f"{row['session_id']} has "
                f"{row['trials']} trials; "
                f"expected {expected_trials}."
            )

            passed = False

        if row["movements"] != len(
            EXPECTED_MOVEMENTS
        ):
            print(
                f"WARNING: "
                f"{row['participant_id']} "
                f"{row['session_id']} has "
                f"{row['movements']} movement classes."
            )

            passed = False

    if passed:
        print(
            "PASS - every session contains "
            f"{expected_trials} trials and "
            f"{len(EXPECTED_MOVEMENTS)} movement classes."
        )

    return passed


def check_movement_distribution(df):
    """Check movement class distribution."""

    print()
    print("=" * 70)
    print("MOVEMENT DISTRIBUTION")
    print("=" * 70)

    trial_table = (
        df[
            [
                "participant_id",
                "session_id",
                "trial_id",
                "movement_label",
            ]
        ]
        .drop_duplicates()
    )

    distribution = (
        trial_table["movement_label"]
        .value_counts()
        .reindex(
            EXPECTED_MOVEMENTS,
            fill_value=0,
        )
    )

    print(distribution.to_string())

    print()

    expected_per_class = (
        len(DATASET_FILES) * 5
    )

    passed = all(
        distribution == expected_per_class
    )

    if passed:
        print(
            "PASS - all movement classes are "
            f"balanced at {expected_per_class} trials each."
        )
    else:
        print(
            "WARNING - movement classes are not balanced."
        )

    return passed


def check_missing_values(df):
    """Check for missing values."""

    print()
    print("=" * 70)
    print("MISSING VALUES")
    print("=" * 70)

    missing = df[
        REQUIRED_COLUMNS
    ].isna().sum()

    missing = missing[
        missing > 0
    ]

    if missing.empty:
        print(
            "PASS - no missing values "
            "in required columns."
        )

        return True

    print(missing.to_string())

    return False


def check_numeric_values(df):
    """Check numerical features for invalid values."""

    print()
    print("=" * 70)
    print("NUMERICAL FEATURE CHECK")
    print("=" * 70)

    passed = True

    for column in [
        "timestamp",
        *FEATURE_COLUMNS,
    ]:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        nan_count = values.isna().sum()

        inf_count = np.isinf(
            values
        ).sum()

        if nan_count > 0:
            print(
                f"{column:<25} "
                f"invalid/non-numeric: {nan_count}"
            )

            passed = False

        if inf_count > 0:
            print(
                f"{column:<25} "
                f"infinite: {inf_count}"
            )

            passed = False

    if passed:
        print(
            "PASS - numerical features contain "
            "no NaN or infinite values."
        )

    return passed


def check_timestamps(df):
    """
    Check that timestamps increase within every trial.
    """

    print()
    print("=" * 70)
    print("TIMESTAMP CHECK")
    print("=" * 70)

    passed = True

    trial_groups = df.groupby(
        [
            "participant_id",
            "session_id",
            "trial_id",
        ]
    )

    for (
        participant,
        session,
        trial,
    ), group in trial_groups:

        timestamps = (
            group["timestamp"]
            .to_numpy()
        )

        if len(timestamps) < 2:
            print(
                f"WARNING: "
                f"{participant}/{session}/{trial} "
                "contains fewer than two observations."
            )

            passed = False
            continue

        differences = np.diff(
            timestamps
        )

        if np.any(differences < 0):
            print(
                f"FAILED: timestamps decrease in "
                f"{participant}/{session}/{trial}"
            )

            passed = False

    if passed:
        print(
            "PASS - timestamps are monotonically "
            "increasing within all trials."
        )

    return passed


def print_trial_lengths(df):
    """Report number of observations per trial."""

    print()
    print("=" * 70)
    print("TRIAL LENGTHS")
    print("=" * 70)

    trial_lengths = (
        df.groupby(
            [
                "participant_id",
                "session_id",
                "trial_id",
                "movement_label",
            ]
        )
        .size()
        .reset_index(
            name="observations"
        )
    )

    print(
        trial_lengths[
            "observations"
        ].describe().to_string()
    )

    print()

    print(
        "Minimum observations in a trial:",
        trial_lengths["observations"].min(),
    )

    print(
        "Maximum observations in a trial:",
        trial_lengths["observations"].max(),
    )

    print(
        "Mean observations per trial:",
        round(
            trial_lengths["observations"].mean(),
            2,
        ),
    )


def print_feature_ranges(df):
    """Print basic ranges for movement features."""

    print()
    print("=" * 70)
    print("FEATURE RANGES")
    print("=" * 70)

    summary = df[
        FEATURE_COLUMNS
    ].describe().T

    summary = summary[
        [
            "mean",
            "std",
            "min",
            "max",
        ]
    ]

    print(
        summary.round(4).to_string()
    )


def main():

    df = load_dataset()

    checks = []

    checks.append(
        check_required_columns(df)
    )

    checks.append(
        check_identifiers(df)
    )

    checks.append(
        check_trial_structure(df)
    )

    checks.append(
        check_movement_distribution(df)
    )

    checks.append(
        check_missing_values(df)
    )

    checks.append(
        check_numeric_values(df)
    )

    checks.append(
        check_timestamps(df)
    )

    print_trial_lengths(df)

    print_feature_ranges(df)

    print()
    print("=" * 70)
    print("FINAL VALIDATION RESULT")
    print("=" * 70)

    if all(checks):
        print(
            "PASS - dataset passed all structural "
            "and numerical validation checks."
        )
    else:
        print(
            "WARNING - one or more validation checks "
            "require attention."
        )


if __name__ == "__main__":
    main()