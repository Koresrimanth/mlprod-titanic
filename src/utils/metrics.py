import json
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class ClassificationMetrics:

    def evaluate(
        self,
        y_true,
        y_pred
    ):

        metrics = {

            "accuracy": float(
                accuracy_score(y_true, y_pred)
            ),

            "precision": float(
                precision_score(y_true, y_pred)
            ),

            "recall": float(
                recall_score(y_true, y_pred)
            ),

            "f1_score": float(
                f1_score(y_true, y_pred)
            ),

            "roc_auc": float(
                roc_auc_score(y_true, y_pred)
            )
        }

        return metrics


    def save_metrics(
        self,
        metrics,
        path
    ):

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        with open(path, "w") as file:

            json.dump(
                metrics,
                file,
                indent=4
            )