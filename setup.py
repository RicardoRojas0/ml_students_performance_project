from setuptools import find_packages, setup

# Function to get the requirements from the requirements.txt file
def get_requirements(file):
    '''
    This function reads the requirements.txt file and returns a list of the requirements
    '''
    # Open the requirements.txt file and read the content into a list of strings
    with open(file) as f:
        requirements = f.read().splitlines()
    
    # Remove the '-e .' string from the list of requirements
        if '-e .' in requirements:
            requirements.remove('-e .')
        
    return requirements

# Setup function to create the package
setup(
    name='ml_students_performance_project',
    version='0.0.1',
    author='Ricardo',
    author_email='ricardorojasm1991@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)