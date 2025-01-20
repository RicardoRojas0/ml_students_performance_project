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
from sklearn.metrics import r2_score

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
            
            params = {
                "Linear Regression": {
                    "fit_intercept": [True, False],
                },
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "absolute_error",
                        "friedman_mse",
                        "poisson",
                    ],
                    "max_depth": [5, 10, 15, 20],
                    "max_features": ["sqrt", "log2"],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 34, 64, 128, 256],
                    "max_features": ["sqrt", "log2"],
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200, 400],
                    "learning_rate": [0.01, 0.1, 1],
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100, 200, 400],
                    "learning_rate": [0.01, 0.1, 1],
                },
                "XGBoost": {
                    "n_estimators": [50, 100, 200, 400],
                    "learning_rate": [0.01, 0.1, 1],
                },
            }

            # Create a dictionary of hyperparameters for the models

            # Evaluate the models using the evaluate_models function
            model_report: dict = evaluate_models(
                models=models,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                param=params,
            )

            # Get the best model based on the R2 Score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # Condition to check if the best model score is greater than 0.6
            if best_model_score < 0.6:
                logging.info("Best model score is less than 0.6")
                raise CustomException("Best model score is less than 0.6")

            logging.info(
                "Best model found on both train and test data based on R2 Score"
            )

            # Save the best model to the disk
            save_object(file_path=self.config.trained_model_path, object=best_model)
            logging.info("Best model saved to the disk")

            # Predict the test data using the best model and calculate the R2 Score
            predicted = best_model.predict(X_test)
            r2_score_result = r2_score(y_test, predicted)

            # Return the best model name and score
            return r2_score_result

        except Exception as e:
            raise CustomException(e, sys)
