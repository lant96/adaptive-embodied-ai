from pathlib import Path
import csv


class FeatureRecorder:
    """Records extracted movement features."""

    def __init__(self, output_file):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.output_file, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "timestamp",
            "shoulder_width",
            "head_offset_x",
            "head_offset_y",
            "torso_dx",
            "torso_dy",
            "movement_speed",
        ])

    def record(self, timestamp, features):
        self.writer.writerow([
            timestamp,
            features["shoulder_width"],
            features["head_offset_x"],
            features["head_offset_y"],
            features["torso_dx"],
            features["torso_dy"],
            features["movement_speed"],
        ])
        self.file.flush()  # so Ctrl+C mid-session doesn't lose the last rows

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()