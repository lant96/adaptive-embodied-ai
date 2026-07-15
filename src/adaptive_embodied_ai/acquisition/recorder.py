from pathlib import Path
import csv


class PoseRecorder:

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

        self.writer = csv.writer(self.file)

        self.writer.writerow(
            [
                "timestamp",
                "landmark",
                "x",
                "y",
                "z",
                "visibility",
            ]
        )

    def record(self, timestamp, landmarks):

        for idx, landmark in enumerate(landmarks):

            self.writer.writerow(
                [
                    timestamp,
                    idx,
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility,
                ]
            )

    def close(self):
        self.file.close()