from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)

# FIXED: Proper CORS configuration for production
CORS(app, resources={
    r"/*": {
        "origins": ["*"],  # In production, replace with your actual frontend URL
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load your trained model and preprocessing objects
try:
    lr_model = joblib.load('lr_mbti_model.joblib')
    scaler = joblib.load('scaler.joblib')
    le_gender = joblib.load('le_gender.joblib')
    le_edu = joblib.load('le_edu.joblib')
    le_interest = joblib.load('le_interest.joblib')
    le_personality = joblib.load('le_personality.joblib')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    # Set to None so we can handle gracefully
    lr_model = None

# MBTI Descriptions
mbti_descriptions = {
    'ENTJ': 'The Commander: Strategic leaders, motivated to organize change',
    'INTJ': 'The Mastermind: Analytical problem-solvers, eager to improve systems and processes',
    'ENTP': 'The Visionary: Inspired innovators, seeking new solutions to challenging problems',
    'INTP': 'The Architect: Philosophical innovators, fascinated by logical analysis',
    'ENFJ': 'The Teacher: Idealist organizers, driven to do what is best for humanity',
    'INFJ': 'The Counselor: Creative nurturers, driven by a strong sense of personal integrity',
    'ENFP': 'The Champion: People-centered creators, motivated by possibilities and potential',
    'INFP': 'The Healer: Imaginative idealists, guided by their own values and beliefs',
    'ESTJ': 'The Supervisor: Hardworking traditionalists, taking charge to get things done',
    'ISTJ': 'The Inspector: Responsible organizers, driven to create order out of chaos',
    'ESFJ': 'The Provider: Conscientious helpers, dedicated to their duties to others',
    'ISFJ': 'The Protector: Industrious caretakers, loyal to traditions and institutions',
    'ESTP': 'The Dynamo: Energetic thrillseekers, ready to push boundaries and dive into action',
    'ISTP': 'The Craftsperson: Observant troubleshooters, solving practical problems',
    'ESFP': 'The Entertainer: Vivacious entertainers, loving life and charming those around them',
    'ISFP': 'The Composer: Gentle caretakers, enjoying the moment with low-key enthusiasm'
}

@app.route('/')
def home():
    return jsonify({
        "message": "MBTI Predictor API is running!",
        "status": "online",
        "model": "Logistic Regression",
        "model_loaded": lr_model is not None,
        "endpoints": {
            "/predict": "POST - Make personality predictions",
            "/model-info": "GET - Get model information"
        }
    })

@app.route('/model-info')
def model_info():
    if lr_model is None:
        return jsonify({
            "success": False,
            "error": "Models not loaded. Please check server logs."
        }), 500
    
    return jsonify({
        "model_type": "Logistic Regression",
        "test_accuracy": "76.88%",
        "train_accuracy": "76.83%",
        "features": [
            "Age", "Gender", "Education", "Introversion Score",
            "Sensing Score", "Thinking Score", "Judging Score", "Interest"
        ],
        "gender_options": le_gender.classes_.tolist(),
        "education_options": le_edu.classes_.tolist(),
        "interest_options": le_interest.classes_.tolist(),
        "personality_types": le_personality.classes_.tolist()
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 200
    
    # Check if models are loaded
    if lr_model is None:
        return jsonify({
            "success": False,
            "error": "Models not loaded on server. Please contact administrator."
        }), 500
    
    try:
        data = request.json
        
        # Validate that we received JSON data
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data received"
            }), 400
        
        # Extract features with validation
        try:
            age = float(data['age'])
        except (KeyError, ValueError, TypeError):
            return jsonify({
                "success": False,
                "error": "Invalid or missing 'age' field"
            }), 400
        
        # Validate required fields
        required_fields = ['gender', 'education', 'interest', 'introversion', 
                          'sensing', 'thinking', 'judging']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400
        
        gender = data['gender']
        education = data['education']
        interest = data['interest']
        
        try:
            introversion = float(data['introversion'])
            sensing = float(data['sensing'])
            thinking = float(data['thinking'])
            judging = float(data['judging'])
        except (ValueError, TypeError):
            return jsonify({
                "success": False,
                "error": "Personality scores must be numeric values"
            }), 400
        
        # Validate score ranges
        for score_name, score_value in [
            ('introversion', introversion), ('sensing', sensing),
            ('thinking', thinking), ('judging', judging)
        ]:
            if not (0 <= score_value <= 10):
                return jsonify({
                    "success": False,
                    "error": f"{score_name} score must be between 0 and 10"
                }), 400
        
        # Validate age range
        if not (10 <= age <= 100):
            return jsonify({
                "success": False,
                "error": "Age must be between 10 and 100"
            }), 400
        
        # Encode categorical features
        try:
            gender_enc = le_gender.transform([gender])[0]
        except ValueError:
            return jsonify({
                "success": False,
                "error": f"Invalid gender. Must be one of: {le_gender.classes_.tolist()}"
            }), 400
        
        try:
            # Handle education as either string or int
            if isinstance(education, str):
                education_enc = le_edu.transform([education])[0]
            else:
                education_enc = int(education)
        except (ValueError, TypeError):
            return jsonify({
                "success": False,
                "error": f"Invalid education. Must be one of: {le_edu.classes_.tolist()} or 0/1"
            }), 400
        
        try:
            interest_enc = le_interest.transform([interest])[0]
        except ValueError:
            return jsonify({
                "success": False,
                "error": f"Invalid interest. Must be one of: {le_interest.classes_.tolist()}"
            }), 400
        
        # Create feature array (order matches training data)
        features = np.array([[
            age, gender_enc, education_enc,
            introversion, sensing, thinking, judging,
            interest_enc
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction using Logistic Regression
        pred_class = lr_model.predict(features_scaled)[0]
        pred_proba = lr_model.predict_proba(features_scaled)[0]
        
        # Get MBTI type
        mbti_type = le_personality.inverse_transform([pred_class])[0]
        
        # Collect probabilities
        probabilities = {}
        for idx, prob in enumerate(pred_proba):
            personality = le_personality.inverse_transform([idx])[0]
            probabilities[personality] = float(prob)
        
        # Get top 5
        top_5 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get description
        description = mbti_descriptions.get(mbti_type, "Description not available")
        
        return jsonify({
            "success": True,
            "predicted_type": mbti_type,
            "description": description,
            "confidence": float(pred_proba[pred_class]),
            "top_5_probabilities": dict(top_5),
            "all_probabilities": probabilities,
            "probabilities": probabilities  # Also include for backwards compatibility
        })
        
    except KeyError as e:
        return jsonify({
            "success": False,
            "error": f"Missing required field: {str(e)}"
        }), 400
    except Exception as e:
        # Log the full error for debugging
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": lr_model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)