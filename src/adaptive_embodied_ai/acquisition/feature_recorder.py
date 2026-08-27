from pathlib import Path
import csv


class FeatureRecorder:
    """Record movement features and metadata to a CSV file."""

    FIELDNAMES = [
        # Experimental metadata
        "participant_id",
        "session_id",
        "trial_id",
        "movement_label",

        # Temporal information
        "timestamp",

        # 2D representation
        "shoulder_width",
        "head_offset_x",
        "head_offset_y",
        "torso_dx",
        "torso_dy",
        "movement_speed",

        # Depth-aware representation
        "head_offset_z",
        "torso_dz",
        "movement_speed_3d",
    ]

    def __init__(self, output_file):
        self.output_file = Path(output_file)

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.file = open(
            self.output_file,
            "w",
            newline="",
        )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.FIELDNAMES,
        )

        self.writer.writeheader()

    def record(self, timestamp, features):
        """Write one feature observation."""

        row = {
            "timestamp": timestamp,
            **features,
        }

        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        """Close the output CSV file."""

        if not self.file.closed:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        self.close()