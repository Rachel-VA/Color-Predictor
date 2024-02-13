"""
this project is created on 12/29/23
VSCode and python virtual environment
Step 1: Open the Integrated Terminal in VS Code
Step 2: Create the Virtual Environment: python -m venv venv
Step 3: Activate the Virtual Environment: .\venv\Scripts\activate
Step 5: Install Required Packages: pip install pandas scikit-learn flask
Note: make sure to select the correct python intepreter for virtual invironment
supervised machine learning model/specifically a classification model
K-Nearest Neighbors (KNN)

installations: pip install pandas scikit-learn flask

to run the app, python app.py, then Ctrl + click to open up the local web
make to to select the correct python environment from VS code (in the bottom)
"""
from flask import Flask, request, render_template
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

app = Flask(__name__)

# Function to load dataset and train model
def train_model():
    data = pd.read_csv('color_data.csv')
    X = data[['R', 'G', 'B']]
    y = data['ColorName']
    model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    model.fit(X, y)
    return model

# Load and train the model
model = train_model()

# Route for handling the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        r = int(request.form['r'])
        g = int(request.form['g'])
        b = int(request.form['b'])
        
        # Ensure input values are within the expected range
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            return render_template('index.html', prediction="Invalid RGB values", r=r, g=g, b=b)
        
        # Create a DataFrame for prediction that includes feature names
        features = pd.DataFrame([[r, g, b]], columns=['R', 'G', 'B'])
        
        # Make a prediction
        prediction = model.predict(features)[0]
        return render_template('index.html', prediction=prediction, r=r, g=g, b=b)
    return render_template('index.html', prediction=None, r=255, g=255, b=255)

if __name__ == '__main__':
    app.run(debug=True)
