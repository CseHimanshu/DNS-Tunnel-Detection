from flask import Flask, render_template, request
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)

# Load the PCA model
loaded_model_pca = pickle.load(open("D:/cdacProject/Logistic_Regression_Model_PCA.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return "No file uploaded"
        
        # Assuming your input data is stored in a variable X
        df = pd.read_csv(file)
        X = df  # Assuming the data is already preprocessed
        
        # Apply PCA transformation
        pca=PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Load the Logistic Regression model
        model_LR = pickle.load(open("Logistic_Regression_Model_PCA.pkl", "rb"))
        
        # Make predictions
        predictions = model_LR.predict(X_pca)
        
        # Convert predictions to a DataFrame
        prediction_results = pd.DataFrame(predictions, columns=['Predictions'])
        
        # Pass prediction results to the HTML template
        return render_template('index.html', tables=[prediction_results.to_html(classes='data', index=False)])

if __name__ == '__main__':
    app.run(debug=True)
