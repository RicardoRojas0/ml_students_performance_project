import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, object):
    """
    Save the object to the disk
    """
    try:
        # Check if the directory exists, if not create the directory
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Directory created at {dir_path}")

        # Save the object to the disk
        with open(file_path, "wb") as file:
            dill.dump(object, file)
        logging.info("Object saved to the disk")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load the object from the disk
    """
    try:
        # Load the object from the disk
        with open(file_path, "rb") as file_object:
            return dill.load(file_object)
        logging.info("Object loaded from the disk")

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(models, X_train, y_train, X_test, y_test, param):
    try:
        # Dictionary to store the evaluation metrics
        report = {}

        # Train and evaluate the models
        for i in range(len(models)):
            model = list(models.values())[i]
            params = param[list(models.keys())[i]]

            # Grid search to find the best hyperparameters
            grid_seach = GridSearchCV(estimator=model, param_grid=params, cv=5)
            grid_seach.fit(X_train, y_train)

            # Train the model
            model.set_params(**grid_seach.best_params_)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate the evaluation metrics
            test_model_score = r2_score(y_test, y_pred)

            # Store the evaluation metrics in the report
            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)
