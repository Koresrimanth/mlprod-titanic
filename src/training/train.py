import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src.preprocessing.preprocessing import (
    DataPreprocessing
)

from src.features.feature_engineering import (
    FeatureEngineering
)

from src.utils.common import (
    save_object,
    read_yaml
)

from src.utils.metrics import (
    ClassificationMetrics
)


config = read_yaml("configs/config.yaml")


class ModelTrainer:

    def train(
        self,
        train_path,
        test_path
    ):

        # Load datasets
        train_df = pd.read_csv(train_path)

        test_df = pd.read_csv(test_path)

        # Feature engineering
        feature_engineering = (
            FeatureEngineering()
        )

        train_df = (
            feature_engineering
            .transform_features(train_df)
        )

        test_df = (
            feature_engineering
            .transform_features(test_df)
        )

        # Split train
        X_train = train_df.drop(
            columns=["Survived"]
        )

        y_train = train_df["Survived"]

        # Split test
        X_test = test_df.drop(
            columns=["Survived"]
        )

        y_test = test_df["Survived"]

        # Preprocessing
        preprocessor = (
            DataPreprocessing()
            .get_preprocessor()
        )

        X_train_processed = (
            preprocessor.fit_transform(X_train)
        )

        X_test_processed = (
            preprocessor.transform(X_test)
        )

        # MLflow tracking
        mlflow.set_tracking_uri(
            "file:./mlruns"
        )

        with mlflow.start_run():

            # Model
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

            # Train
            model.fit(
                X_train_processed,
                y_train
            )

            # Predictions
            predictions = model.predict(
                X_test_processed
            )

            # Metrics
            metrics_obj = (
                ClassificationMetrics()
            )

            metrics = metrics_obj.evaluate(
                y_test,
                predictions
            )

            # Log metrics
            for key, value in metrics.items():

                mlflow.log_metric(
                    key,
                    value
                )

            # Log parameter
            mlflow.log_param(
                "n_estimators",
                100
            )

            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                name="random_forest_model"
            )

        # SAVE ARTIFACTS LOCALLY
        save_object(
            config["artifacts"]["model_path"],
            model
        )

        save_object(
            config["artifacts"]["preprocessor_path"],
            preprocessor
        )

        # SAVE METRICS JSON
        metrics_obj.save_metrics(
            metrics,
            config["artifacts"]["metrics_path"]
        )

        print(
            "Training pipeline completed successfully"
        )