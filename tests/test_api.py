from fastapi.testclient import TestClient

from src.api.app import app


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