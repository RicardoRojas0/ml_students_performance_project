import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import pickle


def save_preprocessor(file_path, preprocessor_obj):
    """
    Save the preprocessor object to the disk
    """
    try:
        # Check if the directory exists, if not create the directory
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Directory created at {dir_path}")

        # Save the preprocessor object to the disk
        with open(file_path, "wb") as file:
            pickle.dump(preprocessor_obj, file)
        logging.info("Preprocessor object saved to the disk")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise CustomException(e, sys)
