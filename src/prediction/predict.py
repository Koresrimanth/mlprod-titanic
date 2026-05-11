import pandas as pd

from src.utils.common import (
    load_object,
    read_yaml
)

from src.features.feature_engineering import (
    FeatureEngineering
)
from src.monitoring.prediction_monitor import (
    PredictionMonitor
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
        self.monitor = PredictionMonitor()

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

        self.monitor.log_prediction(
                        input_data,
                        prediction[0]
                        )
        
        return prediction[0]
    
    def predict_proba(
    self,
    input_data: dict
):

        df = pd.DataFrame([input_data])

        # Feature engineering
        df = (
            self.feature_engineering
            .transform_features(df)
        )

        # Transform
        transformed_data = (
            self.preprocessor.transform(df)
        )

        # Probability
        probability = (
            self.model.predict_proba(
                transformed_data
            )
        )

        return float(probability[0][1])