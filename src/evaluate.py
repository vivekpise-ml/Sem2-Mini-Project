"""
evaluate.py
------------

Updated evaluator aligned with the improved train_classical.py.

Features:
- Flexible label detection and cleaning (works with string labels like
  'phishing','benign','defacement' and numeric labels -1/1/0/1).
- Supports datasets with raw URLs (Option B) and engineered numeric features (Option A).
- Attempts to align labels with model.classes_ when possible.
- Computes accuracy, confusion matrix, classification report, and ROC-AUC
  (binary and multiclass where probabilities/decision_function are available).
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize

from src.features import extract_all_features
from src.config import MODEL_DIR


# -------------------------
# Utilities: numeric helpers
# -------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# -------------------------
# Label detection & cleaning
# -------------------------
def detect_label_column(df, verbose=True):
    """
    Flexible detection of label column. Returns (y_series, label_col_name).
    """
    candidates = [
        "label", "Label", "LABEL",
        "type", "Type", "TYPE",
        "class", "Class", "CLASS",
        "target", "Target",
        "status", "Status",
        "result", "Result",
        "CLASS_LABEL", "Category"
    ]

    for cand in candidates:
        if cand in df.columns:
            if verbose:
                print(f"🔎 Label column detected → {cand}")
            return df[cand].copy(), cand

    # heuristic: column with small number of unique values (not 'url')
    for col in df.columns:
        if col.lower() == "url":
            continue
        nunique = df[col].nunique(dropna=False)
        if nunique <= 50:
            if verbose:
                print(f"ℹ️ Heuristic label column detection → {col} (unique={nunique})")
            return df[col].copy(), col

    raise KeyError("❌ No valid label column found in dataset.")


def clean_label_values(y):
    """
    Convert a pandas Series y to integer labels.
    Returns (y_mapped_series, label_map) where label_map maps original_str -> int.
    """
    # numeric dtype: accept & normalize -1 -> 0
    if pd.api.types.is_numeric_dtype(y):
        y_num = y.replace(-1, 0).astype(int)
        unique_vals = sorted(y_num.unique())
        label_map = {str(k): int(k) for k in unique_vals}
        return y_num, label_map

    # text labels: normalize and map to integers (sorted order)
    y_clean = y.astype(str).str.strip()
    # preserve case-insensitive uniqueness but keep original strings (we'll present them)
    # use lowercase for determining uniqueness to avoid duplicates that differ only by case/whitespace
    lookup = pd.Series(y_clean.values).str.lower().str.strip()
    unique_lower = sorted(lookup.unique())
    # Build map from lower-case value -> int
    label_map = {v: i for i, v in enumerate(unique_lower)}
    # Apply mapping on the original series using lower-case keys
    y_mapped = lookup.map(label_map).astype(int)
    # For convenience return label_map keyed by original-looking strings
    # Also create a human-friendly reverse map
    return y_mapped, label_map


# -------------------------
# Dataset mode detection
# -------------------------
def detect_dataset_mode(df):
    # Option B: Raw URL dataset
    if any(col in df.columns for col in ["url", "URL", "Url"]):
        print("🔎 Detected URL column → Using RAW URL FEATURE EXTRACTION (Option B)")
        return "B"

    # Option A: Many numeric features -> pre-engineered dataset
    numeric_cols = df.select_dtypes(include=["int", "float"]).columns
    if len(numeric_cols) >= 10:
        print("🔎 Detected many numeric columns → Using EXISTING FEATURES (Option A)")
        return "A"

    raise ValueError("❌ Unable to detect dataset structure.")


# -------------------------
# Main evaluation
# -------------------------
def evaluate_models(df):
    print("\n📊 Evaluating saved models from:", MODEL_DIR)

    mode = detect_dataset_mode(df)

    # detect & clean labels
    y_raw, label_col = detect_label_column(df)
    y_mapped, label_map = clean_label_values(y_raw)
    print("🔢 Label mapping used for evaluation (lowercase keys):", label_map)

    # prepare features
    print("🔍 Preparing evaluation feature matrix...")
    if mode == "A":
        print("📊 Using existing numeric columns (Option A)")
        X = df.drop(columns=[label_col, "id"], errors="ignore")
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    else:
        print("🌐 Extracting features from raw URLs (Option B)")
        url_col = next((c for c in ["url", "URL", "Url"] if c in df.columns), None)
        feature_rows = []
        for i, row in df.iterrows():
            try:
                feats = extract_all_features(row[url_col], "")
                feature_rows.append(feats)
            except Exception as e:
                print(f"⚠️ Skipped row {i}: {e}")
                feature_rows.append({})  # keep alignment
        '''
        X = pd.DataFrame(feature_rows).fillna(0)
        '''
        # Build feature matrix
        X = pd.DataFrame(feature_rows)

        # Safety: enforce numeric values
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        # VERY IMPORTANT:
        # Drop URL column from original DF if it leaked in (Option B extractors SHOULD ONLY use features)
        for col in ["url", "URL", "Url"]:
            if col in X.columns:
                X = X.drop(columns=[col], errors="ignore")


    print(f"   ✅ Final evaluation feature shape: {X.shape}")

    results = {}

    # iterate saved models
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

        # load model & scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # scale features (catch mismatches)
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            print(f"❌ Feature mismatch for {model_name}: {e}")
            print("   This model cannot evaluate this dataset format.")
            continue

        # Prepare y for this model.
        # If model exposes classes_, try to align mapping to model.classes_
        y_for_model = y_mapped.copy()
        if hasattr(model, "classes_"):
            model_classes = np.array(model.classes_)
            # if model.classes_ are non-integers (rare), we can't align reliably
            try:
                model_classes_int = model_classes.astype(int)
                set_model = set(model_classes_int.tolist())
                set_y = set(y_for_model.unique().tolist())
                if set_model != set_y:
                    print("⚠️ Mismatch between model.classes_ and evaluation label set.")
                    print(f"    model.classes_ = {model_classes_int}, eval labels = {sorted(set_y)}")
                    # proceed but metrics may be invalid unless the mappings truly match
                else:
                    # they match — ensure y_for_model uses same integer values
                    # nothing to do (y_mapped already integers)
                    pass
            except Exception:
                # model.classes_ not integer-convertible (e.g., strings). Try best-effort mapping:
                print("ℹ️ model.classes_ are not integer-convertible; proceeding with evaluation using mapped integer labels.")
        else:
            print("ℹ️ model does not expose classes_; proceeding with evaluation labels as-mapped.")

        # predictions
        try:
            preds = model.predict(X_scaled)
        except Exception as e:
            print(f"❌ Model {model_name} failed to predict: {e}")
            continue

        # accuracy & confusion & report (use y_for_model)
        try:
            acc = accuracy_score(y_for_model, preds)
        except Exception as e:
            print(f"❌ Error computing accuracy for {model_name}: {e}")
            acc = None

        try:
            cm = confusion_matrix(y_for_model, preds)
        except Exception as e:
            print(f"⚠️ Could not compute confusion matrix: {e}")
            cm = None

        try:
            report = classification_report(y_for_model, preds, output_dict=True)
        except Exception as e:
            print(f"⚠️ Could not compute classification report: {e}")
            report = None

        # ROC-AUC computation
        auc = None
        try:
            # prefer predict_proba
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)
            elif hasattr(model, "decision_function"):
                dec = model.decision_function(X_scaled)
                # decision_function shape: (n,) for binary or (n, n_classes)
                if dec.ndim == 1:
                    proba_pos = sigmoid(dec)
                    proba = np.stack([1 - proba_pos, proba_pos], axis=1)
                else:
                    proba = softmax(dec)
            else:
                proba = None

            # compute AUC properly
            n_classes = len(np.unique(y_for_model))
            if proba is not None:
                if n_classes == 2:
                    # binary
                    # ensure proba shape is (n_samples, 2)
                    if proba.ndim == 1:
                        # unexpected shape
                        proba = np.vstack([1 - proba, proba]).T
                    auc = roc_auc_score(y_for_model, proba[:, 1])
                else:
                    # multiclass: binarize labels then compute macro OVR AUC
                    y_bin = label_binarize(y_for_model, classes=sorted(np.unique(y_for_model)))
                    # if proba has shape (n_samples, n_classes) ok
                    if proba.shape[1] != y_bin.shape[1]:
                        print("⚠️ Probability shape does not match number of classes; skipping ROC-AUC.")
                    else:
                        auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
            else:
                print("⚠️ Model provides no probabilities/decision_function — skipping ROC-AUC.")
        except Exception as e:
            print(f"⚠️ Error computing ROC-AUC for {model_name}: {e}")
            auc = None

        results[model_name] = {
            "accuracy": float(acc) if acc is not None else None,
            "roc_auc": float(auc) if auc is not None else None,
            "report": report,
            "confusion_matrix": cm.tolist() if cm is not None else None
        }

        # summary print
        acc_str = f"{results[model_name]['accuracy']:.4f}" if results[model_name]["accuracy"] is not None else "N/A"
        auc_str = f"{results[model_name]['roc_auc']:.4f}" if results[model_name]["roc_auc"] is not None else "N/A"
        print(f"   🎯 Accuracy: {acc_str}, ROC-AUC: {auc_str}")
        if cm is not None:
            print(f"   📉 Confusion Matrix:\n{cm}")

    return results


# -------------------------
# Standalone execution
# -------------------------
if __name__ == "__main__":
    from src.config import DATA_PATH
    print("\n🚀 Running standalone evaluation from evaluate.py ...")

    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at {DATA_PATH}. Please update config.py.")
    else:
        df = pd.read_csv(DATA_PATH)
        results = evaluate_models(df)

        print("\n📊 Final Evaluation Summary:")
        for name, r in results.items():
            print(f"{name:<20} | Accuracy = {r['accuracy']} | ROC-AUC = {r['roc_auc']}")
