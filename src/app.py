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

# Multiple possible model paths for different environments
model_paths = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl'),  # ../models/model.pkl
    os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl'),            # ../models/model.pkl (alternative)
    os.path.join('models', 'model.pkl'),                                              # models/model.pkl (if running from root)
    os.path.join('..', 'models', 'model.pkl'),                                       # ../models/model.pkl (relative)
]

model = None
model_loaded_path = None

# Try loading model from different paths
for model_path in model_paths:
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            model_loaded_path = model_path
            print(f"✅ Model loaded successfully from {model_path}")
            break
    except Exception as e:
        print(f"❌ Failed to load model from {model_path}: {e}")

if model is None:
    print("❌ Model could not be loaded from any path")
    print("Available files and directories:")
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    print(f"Current directory ({current_dir}): {os.listdir(current_dir) if os.path.exists(current_dir) else 'Not found'}")
    print(f"Parent directory ({parent_dir}): {os.listdir(parent_dir) if os.path.exists(parent_dir) else 'Not found'}")
    models_dir = os.path.join(parent_dir, 'models')
    print(f"Models directory ({models_dir}): {os.listdir(models_dir) if os.path.exists(models_dir) else 'Not found'}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_path": model_loaded_path if model else "No model loaded"
    })

@app.route('/debug')
def debug():
    """Debug endpoint to check file structure"""
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    
    debug_info = {
        "current_dir": current_dir,
        "parent_dir": parent_dir,
        "current_dir_files": os.listdir(current_dir) if os.path.exists(current_dir) else "Not found",
        "parent_dir_files": os.listdir(parent_dir) if os.path.exists(parent_dir) else "Not found",
        "model_paths_tried": model_paths,
        "model_loaded": model is not None,
        "model_loaded_from": model_loaded_path
    }
    
    models_dir = os.path.join(parent_dir, 'models')
    if os.path.exists(models_dir):
        debug_info["models_dir_files"] = os.listdir(models_dir)
    else:
        debug_info["models_dir_files"] = "Models directory not found"
    
    return jsonify(debug_info)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded", "details": "Please check server logs for model loading issues"}), 500
    
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