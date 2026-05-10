from src.prediction.predict import (
    PredictionPipeline
)


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

if prediction == 1:

    print("Passenger Survived")

else:

    print("Passenger Did Not Survive")