from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load your trained model and preprocessing objects
try:
    lr_model = joblib.load('lr_mbti_model.joblib')
    scaler = joblib.load('scaler.joblib')
    le_gender = joblib.load('le_gender.joblib')
    le_edu = joblib.load('le_edu.joblib')
    le_interest = joblib.load('le_interest.joblib')
    le_personality = joblib.load('le_personality.joblib')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

@app.route('/')
def home():
    return jsonify({
        "message": "MBTI Predictor API is running!",
        "endpoints": {
            "/predict": "POST - Make personality predictions"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features from request
        age = float(data['age'])
        gender = data['gender']
        education = data['education']
        interest = data['interest']
        introversion = float(data['introversion'])
        sensing = float(data['sensing'])
        thinking = float(data['thinking'])
        judging = float(data['judging'])
        
        # Encode categorical features
        gender_enc = le_gender.transform([gender])[0]
        education_enc = le_edu.transform([education])[0]
        interest_enc = le_interest.transform([interest])[0]
        
        # Create feature array
        features = np.array([[
            age, gender_enc, education_enc,
            introversion, sensing, thinking, judging,
            interest_enc
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        pred_class = lr_model.predict(features_scaled)[0]
        pred_proba = lr_model.predict_proba(features_scaled)[0]
        
        # Get MBTI type
        mbti_type = le_personality.inverse_transform([pred_class])[0]
        
        # Get all probabilities
        probabilities = {}
        for idx, prob in enumerate(pred_proba):
            personality = le_personality.inverse_transform([idx])[0]
            probabilities[personality] = float(prob)
        
        # Sort by probability and get top 5
        top_5 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return jsonify({
            "success": True,
            "predicted_type": mbti_type,
            "probabilities": dict(top_5),
            "all_probabilities": probabilities
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
