import json
import os

from datetime import datetime


class PredictionMonitor:

    def __init__(self):

        self.log_path = (
            "artifacts/metrics/predictions_log.json"
        )

        os.makedirs(
            os.path.dirname(self.log_path),
            exist_ok=True
        )


    def log_prediction(
        self,
        input_data,
        prediction
    ):

        log_entry = {

            "timestamp": str(
                datetime.now()
            ),

            "input": input_data,

            "prediction": int(prediction)
        }

        # Existing logs
        if os.path.exists(self.log_path):

            with open(
                self.log_path,
                "r"
            ) as file:

                logs = json.load(file)

        else:

            logs = []

        # Append
        logs.append(log_entry)

        # Save logs
        with open(
            self.log_path,
            "w"
        ) as file:

            json.dump(
                logs,
                file,
                indent=4
            )