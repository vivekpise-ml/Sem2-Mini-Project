import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from src.features import extract_features
from src.config import MODEL_DIR

def train_models(df):
    # Convert URLs into feature vectors
    """
    Trains multiple classical ML models and saves them under /models directory.
    """
    print("\n🔍 Extracting features from URLs...")

    feature_rows = []
    for i, row in df.iterrows():
        try:
            url = row.get("url", "")
            html = row.get("html", "")
            feats = extract_all_features(url, html)
            feature_rows.append(feats)
        except Exception as e:
            print(f"⚠️ Skipping row {i}: {e}")
    '''
    X = pd.DataFrame([extract_features(u) for u in df['url']])
    '''
    X = pd.DataFrame(feature_rows)
    y = df['label']

    print(f"✅ Features extracted successfully! Total samples: {len(X)}")
    print(f"Feature columns: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Feature scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
               "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
               "LogisticRegression": LogisticRegression(max_iter=1000),
               "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
             }

    results = {}

    for name, model in models.items():
        print(f"\u25C6 Training {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        # Save model and scaler
        model_path = f"{MODEL_DIR}/{name.lower()}_model.pkl"
        scaler_path = f"{MODEL_DIR}/{name.lower()}_scaler.pkl"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
                                                                                            
        print(f"   ✅ Saved {name} to {model_path}")
        print(f"   ✅ Saved scaler to {scaler_path}")

        # Store performance
        results[name] = {"accuracy": acc, "model_path": model_path, "scaler_path": scaler_path}

    return results

# -------------------------------------------------------------------
# Example usage for standalone testing
# -------------------------------------------------------------------
if __name__ == "__main__":
    from src.config import DATA_PATH
    print("\n🚀 Running standalone training from train_classical.py")
    df = pd.read_csv(DATA_PATH)
    results = train_models(df)
    print("\n📊 Training summary:")
    for name, r in results.items():
        print(f"{name}: Accuracy = {r['accuracy']:.4f}")

