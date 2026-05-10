import os

from src.pipelines.training_pipeline import (
    run_training_pipeline
)

from src.prediction.predict import (
    PredictionPipeline
)


def test_prediction_pipeline():

    # Train model if artifacts missing
    if not os.path.exists(
        "artifacts/models/model.pkl"
    ):

        run_training_pipeline()

    sample_passenger = {

        "Pclass": 3,

        "Sex": "male",

        "Age": 22,

        "SibSp": 1,

        "Parch": 0,

        "Fare": 7.25,

        "Embarked": "S"
    }

    pipeline = PredictionPipeline()

    prediction = pipeline.predict(
        sample_passenger
    )

    assert prediction in [0, 1]