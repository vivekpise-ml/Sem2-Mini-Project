import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

from src.features import extract_all_features
from src.config import MODEL_DIR


# ============================================================
# 1. DETECT DATASET MODE (OPTION A or B)
# ============================================================

def detect_dataset_mode(df):
    """
    Detect:
    - OPTION B → Raw URL dataset (url column exists)
    - OPTION A → Engineered feature dataset (>=10 numeric columns)
    """

    # URL dataset (malicious_phish.csv, phishing dataset, etc.)
    if any(col in df.columns for col in ["url", "URL", "Url"]):
        print("🔎 Detected URL column → Using RAW URL FEATURE EXTRACTION (Option B)")
        return "B"

    # Structured engineered dataset (phishing_legitimate.csv, ARFF)
    numeric_cols = df.select_dtypes(include=["int", "float"]).columns
    if len(numeric_cols) >= 10:
        print("🔎 Detected many numeric columns → Using ENGINEERED FEATURES (Option A)")
        return "A"

    raise ValueError("❌ Unable to detect dataset type.")


# ============================================================
# 2. ROBUST LABEL DETECTION (WORKS FOR ALL DATASETS)
# ============================================================

def detect_label_column(df, verbose=True):
    """
    Works for:
    - malicious_phish.csv (type column)
    - phishing_legitimate.csv (Result, label)
    - ARFF datasets (CLASS_LABEL, class)
    - Any multi-class or binary dataset
    """

    # Preferred label column names
    candidates = [
        "label", "Label", "LABEL",
        "type", "Type", "TYPE",
        "class", "Class", "CLASS",
        "target", "Target",
        "status", "Status",
        "result", "Result",
        "CLASS_LABEL", "Category"
    ]

    # First try matching known names
    for cand in candidates:
        if cand in df.columns:
            print(f"🔎 Label column detected → {cand}")
            y = df[cand].copy()
            return clean_label_values(y), cand

    # Otherwise: detect a column with few unique values (heuristic)
    for col in df.columns:
        nunique = df[col].nunique()
        if nunique <= 50 and col.lower() != "url":
            print(f"ℹ️ Heuristic label column detection → {col}")
            y = df[col].copy()
            return clean_label_values(y), col

    raise KeyError("❌ No valid label column found.")


# ============================================================
# 3. CLEAN LABEL VALUES (STRINGS → INTS)
# ============================================================

def clean_label_values(y):
    """
    Handle:
    - strings (phishing/benign/defacement/malware)
    - numeric -1/1 → convert to 0/1
    """

    # Numeric labels acceptable directly
    if pd.api.types.is_numeric_dtype(y):
        return y.replace(-1, 0).astype(int)

    # Text labels → clean & encode
    y_clean = y.astype(str).str.strip().str.lower()

    unique_vals = sorted(y_clean.unique())
    print("🔢 Label classes found:", unique_vals)

    # Map classes to integers
    label_map = {v: i for i, v in enumerate(unique_vals)}
    print("🔁 Label encoding:", label_map)

    return y_clean.map(label_map).astype(int)


# ============================================================
# 4. TRAINING PIPELINE
# ============================================================

def train_models(df):

    print("\n📘 Detecting dataset structure...")
    mode = detect_dataset_mode(df)

    print("\n📘 Detecting label column...")
    y, label_col = detect_label_column(df)

    # ============================================================
    # OPTION B — RAW URL FEATURE EXTRACTION
    # ============================================================

    if mode == "B":
        print("\n🔍 Extracting features from URLs (Option B)...")

        url_col = next((c for c in ["url", "URL", "Url"] if c in df.columns), None)

        feature_rows = []
        for i, row in df.iterrows():
            try:
                feats = extract_all_features(row[url_col], "")
                feature_rows.append(feats)
            except Exception as e:
                print(f"⚠️ Error on row {i}, skipping → {e}")

        X = pd.DataFrame(feature_rows)

    # ============================================================
    # OPTION A — ENGINEERED FEATURES
    # ============================================================

    else:
        print("\n📊 Using existing numerical features (Option A)...")

        # Remove label columns
        X = df.drop(columns=[label_col], errors="ignore")

        # Remove URL if someone provided both
        X = X.drop(columns=["url", "URL", "Url"], errors="ignore")

    print(f"✅ Final feature set shape: {X.shape}")

    # Convert all to numeric & fill NaNs
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # ============================================================
    # TRAIN–TEST SPLIT (STRATIFIED)
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ============================================================
    # SCALING
    # ============================================================

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ============================================================
    # MODELS
    # ============================================================

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42),

        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", multi_class="auto", random_state=42),

        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\n◆ Training {name} ...")

        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, preds)

        model_path = f"{MODEL_DIR}/{name.lower()}_model.pkl"
        scaler_path = f"{MODEL_DIR}/{name.lower()}_scaler.pkl"

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        print(f"   🎯 Accuracy: {acc:.4f}")
        print(f"   💾 Model saved → {model_path}")

        results[name] = {"accuracy": acc, "model_path": model_path}

    return results


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    from src.config import DATA_PATH
    print("\n🚀 Running standalone training")
    df = pd.read_csv(DATA_PATH)
    results = train_models(df)
    print(results)
