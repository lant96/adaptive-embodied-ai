from pathlib import Path
import csv


class FeatureRecorder:
    """
    Records extracted movement features.
    """

    def __init__(self, output_file):

        self.output_file = Path(output_file)

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        self.file = open(
            self.output_file,
            "w",
            newline=""
        )


        self.writer = csv.writer(
            self.file
        )


        self.writer.writerow(
            [
                "timestamp",
                "torso_x",
                "torso_y",
                "shoulder_width",
                "head_offset_x",
                "head_offset_y",
                "movement_speed",
            ]
        )


    def record(
        self,
        timestamp,
        features
    ):

        self.writer.writerow(
            [
                timestamp,
                features["torso_x"],
                features["torso_y"],
                features["shoulder_width"],
                features["head_offset_x"],
                features["head_offset_y"],
                features["movement_speed"],
            ]
        )


    def close(self):

        self.file.close()