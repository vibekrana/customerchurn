from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os
import sys
import traceback

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
    '/opt/render/project/src/models/model.pkl',                                      # Absolute Render path
    '/opt/render/project/models/model.pkl',                                          # Alternative Render path
]

model = None
model_loaded_path = None
loading_errors = []

# Try loading model from different paths with detailed error logging
for model_path in model_paths:
    try:
        print(f"🔍 Trying to load model from: {model_path}")
        
        # Check if file exists
        if not os.path.exists(model_path):
            error_msg = f"File does not exist: {model_path}"
            print(f"❌ {error_msg}")
            loading_errors.append(error_msg)
            continue
            
        # Check file permissions
        if not os.access(model_path, os.R_OK):
            error_msg = f"File not readable: {model_path}"
            print(f"❌ {error_msg}")
            loading_errors.append(error_msg)
            continue
            
        # Check file size
        file_size = os.path.getsize(model_path)
        print(f"📁 File size: {file_size} bytes")
        
        if file_size == 0:
            error_msg = f"File is empty: {model_path}"
            print(f"❌ {error_msg}")
            loading_errors.append(error_msg)
            continue
        
        # Try to load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Verify model is valid
        if hasattr(model, 'predict') and hasattr(model, 'feature_names_in_'):
            model_loaded_path = model_path
            print(f"✅ Model loaded successfully from {model_path}")
            print(f"✅ Model has {len(model.feature_names_in_)} features")
            break
        else:
            error_msg = f"Invalid model object (missing predict or feature_names_in_): {model_path}"
            print(f"❌ {error_msg}")
            loading_errors.append(error_msg)
            model = None
            
    except Exception as e:
        error_msg = f"Failed to load from {model_path}: {str(e)}"
        print(f"❌ {error_msg}")
        loading_errors.append(error_msg)
        print(f"🔍 Full error: {traceback.format_exc()}")

if model is None:
    print("❌ Model could not be loaded from any path")
    print("📋 All loading errors:")
    for error in loading_errors:
        print(f"   - {error}")
    
    # Additional debugging
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    print(f"\n🔍 Debug info:")
    print(f"Current directory: {current_dir}")
    print(f"Parent directory: {parent_dir}")
    print(f"Working directory: {os.getcwd()}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_path": model_loaded_path if model else "No model loaded",
        "loading_errors": loading_errors[:3],  # Show first 3 errors
        "python_version": sys.version,
        "working_directory": os.getcwd()
    })

@app.route('/debug')
def debug():
    """Debug endpoint to check file structure"""
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    
    debug_info = {
        "current_dir": current_dir,
        "parent_dir": parent_dir,
        "working_dir": os.getcwd(),
        "current_dir_files": os.listdir(current_dir) if os.path.exists(current_dir) else "Not found",
        "parent_dir_files": os.listdir(parent_dir) if os.path.exists(parent_dir) else "Not found",
        "model_paths_tried": model_paths,
        "model_loaded": model is not None,
        "model_loaded_from": model_loaded_path,
        "loading_errors": loading_errors,
        "python_version": sys.version,
        "pickle_protocol": pickle.HIGHEST_PROTOCOL
    }
    
    models_dir = os.path.join(parent_dir, 'models')
    if os.path.exists(models_dir):
        debug_info["models_dir_files"] = os.listdir(models_dir)
        # Check individual file details
        model_file_path = os.path.join(models_dir, 'model.pkl')
        if os.path.exists(model_file_path):
            debug_info["model_file_details"] = {
                "exists": True,
                "size": os.path.getsize(model_file_path),
                "readable": os.access(model_file_path, os.R_OK),
                "absolute_path": os.path.abspath(model_file_path)
            }
    else:
        debug_info["models_dir_files"] = "Models directory not found"
    
    return jsonify(debug_info)

@app.route('/force-reload')
def force_reload():
    """Force reload the model"""
    global model, model_loaded_path, loading_errors
    
    model = None
    model_loaded_path = None
    loading_errors = []
    
    # Try loading again
    for model_path in model_paths:
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                model_loaded_path = model_path
                return jsonify({"success": True, "message": f"Model reloaded from {model_path}"})
        except Exception as e:
            loading_errors.append(f"{model_path}: {str(e)}")
    
    return jsonify({"success": False, "errors": loading_errors})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not loaded", 
            "details": "Please check server logs for model loading issues",
            "loading_errors": loading_errors[:3]
        }), 500
    
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
            "status": "success",
            "model_loaded_from": model_loaded_path
        })
    
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)