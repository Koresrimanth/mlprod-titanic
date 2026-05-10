import pandas as pd

from src.utils.common import (
    load_object,
    read_yaml
)

from src.features.feature_engineering import (
    FeatureEngineering
)


config = read_yaml("configs/config.yaml")


class PredictionPipeline:

    def __init__(self):

        # Load trained model
        self.model = load_object(
            config["artifacts"]["model_path"]
        )

        # Load preprocessor
        self.preprocessor = load_object(
            config["artifacts"]["preprocessor_path"]
        )

        # Feature engineering object
        self.feature_engineering = (
            FeatureEngineering()
        )

    def predict(self, input_data: dict):

        # Convert dict to dataframe
        df = pd.DataFrame([input_data])

        # Feature engineering
        df = self.feature_engineering.transform_features(df)

        # Transform features
        transformed_data = (
            self.preprocessor.transform(df)
        )

        # Prediction
        prediction = self.model.predict(
            transformed_data
        )

        return prediction[0]