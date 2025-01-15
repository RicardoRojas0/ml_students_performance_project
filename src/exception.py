import sys

# Error message detail function
def error_message_detail(error, error_detail:sys):
    # Get the error message and the line number of the error
    _, _, exc_tb = error_detail.exc_info()
    
    # Get the file name of the error
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Error message 
    error_message = f'Error: {error} ocurred in Python file {file_name} at line {exc_tb.tb_lineno}'
   
    return error_message

# Custom exception class
class CustomException(Exception):
    # Constructor
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message) 
        self.error_message = error_message_detail(error_message, error_detail)
    
    # String representation of the error message
    def __str__(self):
        return self.error_message