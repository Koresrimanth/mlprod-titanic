import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src.preprocessing.preprocessing import DataPreprocessing
from src.features.feature_engineering import FeatureEngineering

from src.utils.common import (
    save_object,
    read_yaml
)


config = read_yaml("configs/config.yaml")


class ModelTrainer:

    def train(self, train_path):

        # Read dataset
        df = pd.read_csv(train_path)

        # Feature engineering
        feature_engineering = FeatureEngineering()

        df = feature_engineering.transform_features(df)

        # Split features and target
        X = df.drop(columns=["Survived"])

        y = df["Survived"]

        # Preprocessing
        preprocessor = (
            DataPreprocessing()
            .get_preprocessor()
        )

        X_processed = preprocessor.fit_transform(X)

        # Model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_processed, y)

        # Save model
        save_object(
            config["artifacts"]["model_path"],
            model
        )

        # Save preprocessor
        save_object(
            config["artifacts"]["preprocessor_path"],
            preprocessor
        )

        print("Model training completed")

        return model