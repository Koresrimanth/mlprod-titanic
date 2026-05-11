from fastapi import APIRouter

from src.api.schemas import PassengerInput

from src.prediction.predict import (
    PredictionPipeline
)


router = APIRouter()


@router.get("/")
def home():

    return {
        "message": "Titanic ML API Running"
    }

@router.post("/predict")
def predict(data: PassengerInput):

    # Convert input
    input_data = data.model_dump()

    # Pipeline
    pipeline = PredictionPipeline()

    prediction = pipeline.predict(
        input_data
    )

    probability = pipeline.predict_proba(
        input_data
    )

    # Result
    if prediction == 1:

        result = "Survived"

    else:

        result = "Did Not Survive"

    return {

        "prediction": result,

        "survival_probability": round(
            probability,
            2
        )
    }