To configure this project to work with NumPy (if NumPy is not already installed on your computer, or if it's a different version), run:
python -m venv venv
This will create virtual environment, then:
For windows run:
venv\Scripts\activate
For mac run:
source venv/bin/activate
Then run:
pip install -r requirements.txt
You are now good to run the main project. When you're done, run:
deactivate
To deactivate the virtual environment