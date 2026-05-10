from src.ingestion.data_ingestion import DataIngestion
from src.training.train import ModelTrainer

def run_training_pipeline():

    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion()

    trainer = ModelTrainer()

    trainer.train(train_path)

    print("Training pipeline completed successfully")