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

    # Convert Pydantic model to dict
    input_data = data.dict()

    # Prediction pipeline
    pipeline = PredictionPipeline()

    prediction = pipeline.predict(
        input_data
    )

    # Return prediction
    if prediction == 1:

        result = "Survived"

    else:

        result = "Did Not Survive"

    return {
        "prediction": result
    }