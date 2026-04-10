from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Import the separated, fast inference logic from the main ML script
from main import predict_disease, get_available_symptoms, init_inference_model

app = Flask(__name__)
# Automatically opens up CORS so external web apps can connect if needed
CORS(app)

# Initialize the model singleton in memory when the API server boots
init_inference_model(quick=False)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'Invalid format. Expected JSON with "symptoms" key.'}), 400
            
        symptoms = data['symptoms']
        
        if not isinstance(symptoms, list):
            return jsonify({'error': '"symptoms" must be a JSON array of symptom strings.'}), 400
            
        # Call the fast standalone prediction function from main.py
        result = predict_disease(symptoms)
        
        # Merge result with success status
        return jsonify({
            **result,
            'status': 'success'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Helper endpoint to fetch all available valid symptom strings."""
    symptoms = get_available_symptoms()
    return jsonify({
        'symptoms': symptoms,
        'count': len(symptoms)
    })

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Rare Disease MAML API on port {port}")
    app.run(host='0.0.0.0', port=port)
