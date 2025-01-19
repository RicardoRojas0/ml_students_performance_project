import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)


# Data ingestion configuration class to store the data paths
@dataclass
class DataIngestionConfig:
    data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


# Data ingestion class to load the data and split it into train and test data
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_data(self):
        logging.info("Initiating data ingestion")
        try:
            # Loading the data from the data path
            data = pd.read_csv("src/notebook/data/students.csv")
            logging.info(f"Data loaded successfully from {self.config.data_path}")

            # Creating folder for train and test data if it does not exist
            os.makedirs(os.path.dirname(self.config.data_path), exist_ok=True)

            # Saving the data to the data path
            data.to_csv(self.config.data_path, index=False, header=True)

            # Splitting the data into train and test data
            logging.info("Splitting data into train and test data")
            train_data, test_data = train_test_split(
                data, test_size=0.2, random_state=42
            )

            # Saving the train and test data to the respective paths
            train_data.to_csv(self.config.train_data_path, index=False, header=True)
            test_data.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return (
                self.config.train_data_path,
                self.config.test_data_path,
            )
        except Exception as e:
            logging.error("Error loading data")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Load the data and split it into train and test data
    config = DataIngestionConfig()
    data_ingestion = DataIngestion(config)
    train_data, test_data = data_ingestion.load_data()

    # Initiate the data transformation process
    data_transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation(data_transformation_config)
    data_transformation.initiate_data_transformation(train_data, test_data)
