import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pickle

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_data

def train_model():
    # --- Load and preprocess data ---
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found at {data_path}")
        return
    
    print(f"📊 Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} rows")
    
    df = preprocess_data(df)
    print(f"✅ Data preprocessed, shape: {df.shape}")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    print(f"📈 Features: {X.shape[1]}")
    print(f"📊 Target distribution: {y.value_counts().to_dict()}")

    # --- Split and balance dataset ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"✅ Train/test split: {X_train.shape[0]}/{X_test.shape[0]}")
    
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    print(f"✅ SMOTE applied: {X_train_bal.shape[0]} balanced samples")

    # --- Train Smaller Random Forest (for deployment) ---
    print("🚀 Training optimized Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=50,     # Reduced from 150
        max_depth=8,         # Reduced from 10
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_bal, y_train_bal)
    print("✅ Model training completed")

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))
    
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Create models directory if it doesn't exist ---
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    # --- Save model ---
    model_path = os.path.join(models_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Check file size
    file_size = os.path.getsize(model_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"✅ Model saved to {model_path}")
    print(f"📁 Model file size: {file_size_mb:.2f} MB")
    
    if file_size_mb > 100:
        print("⚠️ WARNING: Model file is over 100MB - GitHub may reject it")
    else:
        print("✅ Model file size is acceptable for GitHub")

    # --- Plot feature importance ---
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:15]
    
    plt.figure(figsize=(10, 8))
    feat_imp.plot(kind="barh")
    plt.title("Top 15 Features - Random Forest")
    plt.xlabel("Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = os.path.join(models_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Feature importance plot saved to {plot_path}")
    
    # --- Save feature names for reference ---
    feature_names_path = os.path.join(models_dir, "feature_names.txt")
    with open(feature_names_path, "w") as f:
        for feature in X.columns:
            f.write(feature + "\n")
    print(f"📝 Feature names saved to {feature_names_path}")
    
    return model, X_test, y_test, y_pred

if __name__ == "__main__":
    train_model()