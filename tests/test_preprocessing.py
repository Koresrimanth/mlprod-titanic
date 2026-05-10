import pandas as pd

from src.preprocessing.preprocessing import (
    DataPreprocessing
)


def test_preprocessing_pipeline():

    sample_data = pd.DataFrame({

        "Age": [22],

        "Fare": [7.25],

        "SibSp": [1],

        "Parch": [0],

        "FamilySize": [1],

        "Sex": ["male"],

        "Embarked": ["S"],

        "Pclass": [3],

        "IsAlone": [0]
    })

    preprocessor = (
        DataPreprocessing()
        .get_preprocessor()
    )

    transformed_data = (
        preprocessor.fit_transform(sample_data)
    )

    # Check transformed rows
    assert transformed_data.shape[0] == 1

    # Check transformed columns exist
    assert transformed_data.shape[1] > 0