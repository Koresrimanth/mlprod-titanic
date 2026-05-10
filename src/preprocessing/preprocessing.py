from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


class DataPreprocessing:

    def get_preprocessor(self):

        numeric_features = [
            "Age",
            "Fare",
            "SibSp",
            "Parch",
            "FamilySize"
        ]

        categorical_features = [
            "Sex",
            "Embarked",
            "Pclass",
            "IsAlone"
        ]

        # Numeric pipeline
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),

                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        # Categorical pipeline
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),

                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]
        )

        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric_pipeline",
                    numeric_pipeline,
                    numeric_features
                ),

                (
                    "categorical_pipeline",
                    categorical_pipeline,
                    categorical_features
                )
            ]
        )

        return preprocessor