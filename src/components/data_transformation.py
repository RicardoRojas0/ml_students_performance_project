# System imports
import sys
import os
from dataclasses import dataclass

# Third-party imports
import numpy as np
import pandas as pd

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Custom imports
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        """
        Constructor to initialize the DataTransformation class
        """
        self.config = config

    def data_transformer(self):
        """
        Data transformer method is used to transform the data using the preprocessor
        """
        try:
            # Create lists of numerical and categorical features
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            logging.info("Numerical and categorical features created")
            logging.info(f"Numerical features: {numerical_features}")
            logging.info(f"Categorical features: {categorical_features}")

            # Create a pipeline for numerical features with imputer and scaler
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            logging.info("Numerical features pipeline created and standardized")

            # Create a pipeline for categorical features with imputer, encoder and scaler
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info("Categorical features pipeline created and encoded")

            # Create a preprocessor with numerical and categorical pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical", numerical_pipeline, numerical_features),
                    ("categorical", categorical_pipeline, categorical_features),
                ]
            )

            return preprocessor
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        """
        This method is used to initiate the data transformation process
        """
        try:
            # Load the train and test data
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            logging.info("Train and test data loaded successfully")

            # Transform the data using the preprocessor
            preprocessor = self.data_transformer()
            logging.info("Data transformation initiated successfully")

            # Define the target feature
            target_feature = "math_score"

            # Create dependent and independent variables for train and test data
            train_features = train_data.drop(target_feature, axis=1)
            train_target_feature = train_data[target_feature]

            test_features = test_data.drop(target_feature, axis=1)
            test_target_feature = test_data[target_feature]
            logging.info("Dependent and independent variables created successfully")

            # Fit the preprocessor on the train data and transform the train and test features
            train_features_transformed = preprocessor.fit_transform(train_features)
            test_features_transformed = preprocessor.transform(test_features)
            logging.info(
                "Preprocessor fitted on the train data and train and test features transformed successfully"
            )

            # Concatenate the transformed features with the target feature
            train_array = np.c_[
                train_features_transformed, np.array(train_target_feature)
            ]
            test_array = np.c_[test_features_transformed, np.array(test_target_feature)]
            logging.info(
                "Transformed features concatenated with the target feature successfully"
            )

            save_object(
                file_path=self.config.preprocessor_path,
                object=preprocessor,
            )
            logging.info("Preprocessor saved successfully")

            return (
                train_array,
                test_array,
                self.config.preprocessor_path,
            )
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)
