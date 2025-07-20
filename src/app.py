from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os
import sys

# Add the src directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_data
from flask_cors import CORS

# Create Flask app with correct paths
app = Flask(__name__, 
           static_folder="../static", 
           template_folder="../templates")
CORS(app)

# Load model with correct path
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"❌ Model file not found at {model_path}")
    model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        input_df = pd.DataFrame([data])
        processed_df = preprocess_data(input_df)

        # Ensure all required features are present
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in processed_df.columns:
                processed_df[col] = 0
        
        # Reorder columns to match model training
        processed_df = processed_df[model_features]

        prediction = model.predict(processed_df)[0]
        confidence = model.predict_proba(processed_df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "confidence": round(float(confidence), 3),
            "churn_probability": round(float(confidence), 3),
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)