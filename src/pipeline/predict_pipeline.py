import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


# Define the PredictPipeline class
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, data):
        """
        This method makes a prediction using the model
        """
        try:
            # Get the model and preprocessor path
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            # Load the model
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Scale the data
            data_scaled = preprocessor.transform(data)

            # Make prediction
            prediction = model.predict(data_scaled)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)


# Define the predict method
class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        writing_score: int,
        reading_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.writing_score = writing_score
        self.reading_score = reading_score

    def get_data_as_data_frame(self):
        """
        This method converts the data to a DataFrame
        """
        try:
            # Create a dictionary with the data
            data_input = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "writing_score": [self.writing_score],
                "reading_score": [self.reading_score],
            }

            return pd.DataFrame(data_input)

        except Exception as e:
            raise CustomException(e, sys)
