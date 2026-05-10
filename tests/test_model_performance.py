import os
import pandas as pd

from sklearn.metrics import accuracy_score

from src.pipelines.training_pipeline import (
    run_training_pipeline
)

from src.utils.common import (
    load_object,
    read_yaml
)

from src.features.feature_engineering import (
    FeatureEngineering
)


config = read_yaml("configs/config.yaml")


def ensure_model_exists():

    if not os.path.exists(
        "artifacts/models/model.pkl"
    ):

        run_training_pipeline()


def test_model_accuracy():

    ensure_model_exists()

    # Load test dataset
    test_df = pd.read_csv(
        config["artifacts"]["test_data_path"]
    )

    # Feature engineering
    feature_engineering = (
        FeatureEngineering()
    )

    test_df = (
        feature_engineering
        .transform_features(test_df)
    )

    # Split features and target
    X_test = test_df.drop(
        columns=["Survived"]
    )

    y_test = test_df["Survived"]

    # Load preprocessor
    preprocessor = load_object(
        config["artifacts"]["preprocessor_path"]
    )

    X_test_processed = (
        preprocessor.transform(X_test)
    )

    # Load model
    model = load_object(
        config["artifacts"]["model_path"]
    )

    predictions = model.predict(
        X_test_processed
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # Minimum expected accuracy
    assert accuracy > 0.70