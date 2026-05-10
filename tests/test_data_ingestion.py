import os

from src.ingestion.data_ingestion import (
    DataIngestion
)


def test_data_ingestion():

    ingestion = DataIngestion()

    train_path, test_path = (
        ingestion.initiate_data_ingestion()
    )

    assert os.path.exists(train_path)

    assert os.path.exists(test_path)