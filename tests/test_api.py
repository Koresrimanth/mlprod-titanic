import os

from fastapi.testclient import TestClient

from src.api.app import app

from src.pipelines.training_pipeline import (
    run_training_pipeline
)


# Ensure artifacts exist
if not os.path.exists(
    "artifacts/models/model.pkl"
):

    run_training_pipeline()


client = TestClient(app)


def test_home_route():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Titanic ML API Running"
    }


def test_predict_route():

    response = client.post(
        "/predict",

        json={

            "Pclass": 3,

            "Sex": "male",

            "Age": 22,

            "SibSp": 1,

            "Parch": 0,

            "Fare": 7.25,

            "Embarked": "S"
        }
    )

    assert response.status_code == 200

    assert "prediction" in response.json()
    