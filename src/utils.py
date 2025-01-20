import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

        # Save the preprocessor object to the disk
        with open(file_path, "wb") as file:
            dill.dump(object, file)
        logging.info("Preprocessor object saved to the disk")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise CustomException(e, sys)


def evaluate_models(models, X_train, y_train, X_test, y_test, params):
    # Dictionary to store the evaluation metrics
    report = {}

    # Train and evaluate the models
    for model_name, model in models.items():
        # Get the hyperparameters for the current model
        param_grid = params.get(model_name, {})

        # Perform GridSearchCV to find the best hyperparameters
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=5, scoring="r2", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Get the best model from GridSearchCV
        best_model = grid_search.best_estimator_

        # Predict using the best model
        y_pred = best_model.predict(X_test)

        report[model_name] = {
            "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
            "Mean Squared Error": mean_squared_error(y_test, y_pred),
            "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2 Score": r2_score(y_test, y_pred),
            "Adjusted R2 Score": 1
            - (1 - r2_score(y_test, y_pred))
            * (len(y_test) - 1)
            / (len(y_test) - X_test.shape[1] - 1),
        }

    return report
