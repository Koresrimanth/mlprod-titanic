import os

from src.pipelines.training_pipeline import (
    run_training_pipeline
)

from src.prediction.predict import (
    PredictionPipeline
)


def ensure_model_exists():

    if not os.path.exists(
        "artifacts/models/model.pkl"
    ):

        run_training_pipeline()


def test_high_survival_passenger():

    ensure_model_exists()

    passenger = {

        "Pclass": 1,

        "Sex": "female",

        "Age": 30,

        "SibSp": 0,

        "Parch": 0,

        "Fare": 100,

        "Embarked": "C"
    }

    pipeline = PredictionPipeline()

    prediction = pipeline.predict(
        passenger
    )

    assert prediction == 1


def test_low_survival_passenger():

    ensure_model_exists()

    passenger = {

        "Pclass": 3,

        "Sex": "male",

        "Age": 50,

        "SibSp": 0,

        "Parch": 0,

        "Fare": 7.25,

        "Embarked": "S"
    }

    pipeline = PredictionPipeline()

    prediction = pipeline.predict(
        passenger
    )

    assert prediction == 0


def test_missing_age_prediction():

    ensure_model_exists()

    passenger = {

        "Pclass": 2,

        "Sex": "male",

        "Age": None,

        "SibSp": 0,

        "Parch": 0,

        "Fare": 20,

        "Embarked": "S"
    }

    pipeline = PredictionPipeline()

    prediction = pipeline.predict(
        passenger
    )

    assert prediction in [0, 1]
    