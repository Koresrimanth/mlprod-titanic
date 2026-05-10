import os
import pandas as pd

from sklearn.model_selection import train_test_split

from src.utils.common import read_yaml
from src.utils.logger import logging


config = read_yaml("configs/config.yaml")


class DataIngestion:

    def initiate_data_ingestion(self):

        source_path = config["data"]["source_data_path"]

        raw_path = config["data"]["raw_data_path"]

        train_path = config["artifacts"]["train_data_path"]

        test_path = config["artifacts"]["test_data_path"]

        df = pd.read_csv(source_path)

        os.makedirs(os.path.dirname(raw_path), exist_ok=True)

        os.makedirs(os.path.dirname(train_path), exist_ok=True)

        df.to_csv(raw_path, index=False)

        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        train_set.to_csv(train_path, index=False)

        test_set.to_csv(test_path, index=False)

        logging.info("Data ingestion completed")

        return train_path, test_path