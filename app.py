import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained KNN model
model = joblib.load('breastcancer_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values for the 7 features
    features = [float(request.form.get(f)) for f in ['radius_mean', 'texture_mean', 'perimeter_mean', 
                                                     'area_mean', 'smoothness_mean', 'compactness_mean', 
                                                     'concavity_mean']]
    final_features = np.array([features])

    # Make prediction
    prediction = model.predict(final_features)

    # Optional: Get prediction probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(final_features)[:, 1][0]
        result = f"Breast cancer detected with a probability of {probability:.2f}." if prediction[0] == 1 else f"No breast cancer detected with a probability of {probability:.2f}."
    else:
        result = f"Breast cancer detected." if prediction[0] == 1 else f"No breast cancer detected."

    return render_template('result.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
