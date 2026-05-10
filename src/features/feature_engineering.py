import pandas as pd


class FeatureEngineering:

    def transform_features(self, df: pd.DataFrame):

        # Create FamilySize feature
        df["FamilySize"] = df["SibSp"] + df["Parch"]

        # Create IsAlone feature
        df["IsAlone"] = (df["FamilySize"] == 0).astype(int)

        # Drop unnecessary columns
        columns_to_drop = [
            "PassengerId",
            "Name",
            "Ticket",
            "Cabin"
        ]

        df.drop(
            columns=columns_to_drop,
            inplace=True,
            errors="ignore"
        )

        return df