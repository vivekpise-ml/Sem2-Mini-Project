import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from src.features import extract_all_features
from src.config import MODEL_DIR



def detect_dataset_mode(df):
    """
    Automatically detect whether to use:
    - OPTION B: URL feature extraction (raw URLs)
    - OPTION A: Existing numeric features (pre-engineered dataset)
    """

    # If URL column exists → Raw URL dataset → Extract features
    url_cols = ["url", "URL", "Url"]
    if any(col in df.columns for col in url_cols):
        print("🔎 Detected URL column → Using RAW URL FEATURE EXTRACTION (Option B)")
        return "B"

    # If dataset has >10 numeric columns → Kaggle/UCI engineered dataset
    numeric_cols = df.select_dtypes(include=["int", "float"]).columns
    if len(numeric_cols) >= 10:
        print("🔎 Detected many numeric columns → Using EXISTING FEATURES (Option A)")
        return "A"

    raise ValueError("❌ Unable to detect dataset format (no URL column, not enough numeric columns).")


def detect_label_column(df):
    """
    Automatically find correct label column in any dataset.
    """
    candidates = ["CLASS_LABEL", "Label", "label", "Result", "status", "Phishing"]
    for col in candidates:
        if col in df.columns:
            print(f"🔎 Label column detected → {col}")
            return df[col]

    raise KeyError("❌ No valid label column found in dataset.")



def train_models(df):

    print("\n📘 Detecting dataset structure...")
    mode = detect_dataset_mode(df)

    print("\n📘 Detecting label column...")
    y = detect_label_column(df)

    # ------------------------------------------------------------
    # ENCODE LABEL
    # ------------------------------------------------------------
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)
    else:
        y = y.replace(-1, 0)   # Convert -1/1 → 0/1

    # ------------------------------------------------------------
    # OPTION B — Extract features from RAW URL dataset
    # ------------------------------------------------------------
    if mode == "B":
        print("\n🔍 Extracting features from URLs (Option B)...")

        feature_rows = []
        url_col = None

        for col in ["url", "URL", "Url"]:
            if col in df.columns:
                url_col = col
                break

        for i, row in df.iterrows():
            try:
                feats = extract_all_features(row[url_col], "")
                feature_rows.append(feats)
            except Exception as e:
                print(f"⚠️ Skipping row {i}: {e}")

        X = pd.DataFrame(feature_rows)

    # ------------------------------------------------------------
    # OPTION A — Use existing numerical features directly
    # ------------------------------------------------------------
    else:
        print("\n📊 Using existing numerical features (Option A)...")

        # Remove label + non-useful columns
        X = df.drop(columns=["CLASS_LABEL", "Label", "label", "Result", "status",
                             "id"], errors="ignore")

    print(f"✅ Final feature set shape: {X.shape}")

    # Clean numeric features
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # ------------------------------------------------------------
    # Train-test split (important: stratify=y)
    # ------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------
    # SCALE
    # ------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------
    # MODELS
    # ------------------------------------------------------------
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"\n◆ Training {name} ...")

        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)

        # Save model + scaler
        model_path = f"{MODEL_DIR}/{name.lower()}_model.pkl"
        scaler_path = f"{MODEL_DIR}/{name.lower()}_scaler.pkl"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        print(f"   🎯 Accuracy: {acc:.4f}")
        print(f"   💾 Model saved: {model_path}")

        results[name] = {"accuracy": acc}

    return results



if __name__ == "__main__":
    from src.config import DATA_PATH
    print("\n🚀 Running standalone training")
    df = pd.read_csv(DATA_PATH)
    results = train_models(df)
    print(results)
