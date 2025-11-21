"""
evaluate.py
------------
Evaluates trained phishing detection models and compares their performance.

Steps:
1. Loads saved models and scalers from MODEL_DIR
2. Automatically detects whether dataset uses precomputed numeric features (Option A)
   OR raw URLs requiring feature extraction (Option B)
3. Extracts or selects the correct features accordingly
4. Transforms features using each model's saved scaler
5. Evaluates accuracy, ROC-AUC, and prints classification reports
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix

from src.features import extract_all_features
from src.config import MODEL_DIR


# -------------------------------------------------------------------
# Utility: detect label column (keeps your df['label'] logic)
# -------------------------------------------------------------------
def detect_label_column(df):
    for col in ["CLASS_LABEL", "Label", "label", "Result", "status"]:
        if col in df.columns:
            return col
    raise KeyError("❌ No label column found in dataset.")


# -------------------------------------------------------------------
# Utility: detect dataset mode (Option A / Option B)
# -------------------------------------------------------------------
def detect_dataset_mode(df):
    # Option B: Raw URL dataset
    if any(col in df.columns for col in ["url", "URL", "Url"]):
        print("🔎 Detected URL column → Using RAW URL FEATURE EXTRACTION (Option B)")
        return "B"

    # Option A: Many numeric features -> pre-engineered dataset (Kaggle / UCI)
    numeric_cols = df.select_dtypes(include=["int", "float"]).columns
    if len(numeric_cols) >= 10:
        print("🔎 Detected many numeric columns → Using EXISTING FEATURES (Option A)")
        return "A"

    raise ValueError("❌ Unable to detect dataset structure.")


# -------------------------------------------------------------------
# Main evaluation function
# -------------------------------------------------------------------
def evaluate_models(df):
    print("\n📊 Evaluating saved models from:", MODEL_DIR)

    # --------------------------------------
    # Detect correct dataset mode (A or B)
    # --------------------------------------
    mode = detect_dataset_mode(df)

    # --------------------------------------
    # Identify label column
    # --------------------------------------
    label_col = detect_label_column(df)
    y_true = df[label_col]

    # --------------------------------------
    # Generate the feature matrix X
    # --------------------------------------
    print("🔍 Preparing evaluation feature matrix...")

    if mode == "A":
        # Kaggle/UCI engineered dataset
        print("📊 Using existing numeric feature columns (Option A)")
        X = df.drop(columns=[
            label_col, "id"
        ], errors="ignore")

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    else:
        # Raw URL dataset with URL strings
        print("🌐 Extracting features from raw URLs (Option B)")
        feature_rows = []
        url_col = None

        for c in ["url", "URL", "Url"]:
            if c in df.columns:
                url_col = c
                break

        for i, row in df.iterrows():
            try:
                feats = extract_all_features(row[url_col], "")
                feature_rows.append(feats)
            except Exception as e:
                print(f"⚠️ Skipped row {i}: {e}")
                feature_rows.append({})  # keep row count aligned

        X = pd.DataFrame(feature_rows).fillna(0)

    print(f"   ✅ Final evaluation feature shape: {X.shape}")

    # Collect final results
    results = {}

    # --------------------------------------
    # Load all saved models & scalers
    # --------------------------------------
    for file in os.listdir(MODEL_DIR):
        if not file.endswith("_model.pkl"):
            continue

        model_name = file.replace("_model.pkl", "")
        model_path = os.path.join(MODEL_DIR, file)
        scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.pkl")

        if not os.path.exists(scaler_path):
            print(f"⚠️ Skipping {model_name} (scaler missing).")
            continue

        print(f"\n🧠 Evaluating model: {model_name}")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Apply scaler safely
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            print(f"❌ Feature mismatch for {model_name}: {e}")
            print("   This model cannot evaluate this dataset format.")
            continue

        preds = model.predict(X_scaled)

        # Metrics
        acc = accuracy_score(y_true, preds)
        try:
            auc = roc_auc_score(y_true, preds)
        except:
            auc = 0.0

        report = classification_report(y_true, preds, output_dict=True)
        cm = confusion_matrix(y_true, preds)

        results[model_name] = {
            "accuracy": acc,
            "roc_auc": auc,
            "report": report,
            "confusion_matrix": cm.tolist(),
        }

        # Print summary
        print(f"   🎯 Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")
        print(f"   📉 Confusion Matrix: {cm.tolist()}")

    return results


# -------------------------------------------------------------------
# Standalone test
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Running standalone evaluation from evaluate.py ...")

    from src.config import DATA_PATH

    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at {DATA_PATH}. Please update config.py.")
    else:
        df = pd.read_csv(DATA_PATH)
        results = evaluate_models(df)

        print("\n📊 Final Evaluation Summary:")
        for name, r in results.items():
            print(f"{name:<20} | Accuracy = {r['accuracy']:.4f} | ROC-AUC = {r['roc_auc']:.4f}")
