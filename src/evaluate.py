"""
evaluate.py
------------
Evaluates trained phishing detection models and compares their performance.

Steps:
1. Loads saved models and scalers from MODEL_DIR
2. Extracts URL + HTML features from DATA_PATH
3. Transforms features using each model’s scaler
4. Evaluates accuracy, ROC-AUC, and prints classification reports
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix

from src.features import extract_all_features
from src.config import MODEL_DIR, DATA_PATH


def evaluate_models(df):
    print("\n📊 Evaluating saved models from:", MODEL_DIR)

    # Prepare dataset → convert URLs into feature matrix
    print("🔍 Extracting features from dataset...")
    feature_rows = []
    for i, row in df.iterrows():
        try:
            url = row.get("url", "")
            html = row.get("html", "")
            feats = extract_all_features(url, html)
            feature_rows.append(feats)
        except Exception as e:
            print(f"⚠️ Skipping row {i} due to: {e}")

    X = pd.DataFrame(feature_rows)
    y_true = df["label"]

    # Collect results
    results = {}

    # Loop over all model files found in MODEL_DIR
    for file in os.listdir(MODEL_DIR):
        if not file.endswith("_model.pkl"):
            continue

        model_name = file.replace("_model.pkl", "")
        model_path = os.path.join(MODEL_DIR, file)
        scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.pkl")

        if not os.path.exists(scaler_path):
            print(f"⚠️ Skipping {model_name} — missing scaler file.")
            continue

        print(f"\n🧠 Evaluating model: {model_name}")

        # Load model + scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Scale features
        X_scaled = scaler.transform(X)

        # Predict
        preds = model.predict(X_scaled)

        # Metrics
        acc = accuracy_score(y_true, preds)
        auc = roc_auc_score(y_true, preds)
        report = classification_report(y_true, preds, output_dict=True)
        cm = confusion_matrix(y_true, preds)

        results[model_name] = {
            "accuracy": acc,
            "roc_auc": auc,
            "report": report,
            "confusion_matrix": cm.tolist(),
        }

        # Print concise summary
        print(f"   ✅ Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")
        print(f"   📉 Confusion Matrix: {cm.tolist()}")

    return results


# -------------------------------------------------------------------
# Standalone test
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Running standalone evaluation from evaluate.py ...")

    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at {DATA_PATH}. Please update config.py.")
    else:
        df = pd.read_csv(DATA_PATH)
        results = evaluate_models(df)

        print("\n📊 Final Evaluation Summary:")
        for model_name, r in results.items():
            print(f"{model_name:<20} | Accuracy = {r['accuracy']:.4f} | ROC-AUC = {r['roc_auc']:.4f}")
