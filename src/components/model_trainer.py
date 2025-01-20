# System imports
import os
import sys
from dataclasses import dataclass

# Modelling imports - Leaving out CatBoost for problems with Pyhon 3.13
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Custom imports
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model = None

    def initiate_model_trainer(self, train_array, test_array):
        """
        This function initiates the model training process
        """
        try:
            logging.info("Initiating model training process")
            logging.info(
                "Split data into train and test, and dependent and independent variables"
            )

            # Split data into train and test, and dependent and independent variables
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],  # Independent variables for training
                test_array[:, :-1],  # Independent variables for testing
                train_array[:, -1],  # Dependent variable for training
                test_array[:, -1],  # Dependent variable for testing
            )

            # Create a dictionary of models to be trained
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "KNN": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
            }

            # Evaluate the models using the evaluate_models function
            model_report: dict = evaluate_models(
                models=models,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

            # Get the best model name and score
            best_model_name = max(
                model_report, key=lambda x: model_report[x]["R2 Score"]
            )
            logging.info(f"Best model: {best_model_name}")

            best_model_score = model_report[best_model_name]["R2 Score"]
            logging.info(f"Best model score: {best_model_score}")

            # Save the best model to the disk
            best_model = models[best_model_name]

            # Condition to check if the best model score is greater than 0.6
            if best_model_score < 0.6:
                logging.info("Best model score is less than 0.6")
                raise CustomException("Best model score is less than 0.6", sys)

            logging.info(
                "Best model found on both train and test data based on R2 Score"
            )

            # Save the best model to the disk
            save_object(file_path=self.config.trained_model_path, object=best_model)
            logging.info("Best model saved to the disk")

            # Return the best model name and score
            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)
