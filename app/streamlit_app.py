import sys
import os
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Ensure project root is in PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.features import extract_all_features
from src.charcnn_predict import load_charcnn, predict_single, predict_urls


# --------------------------
# MODEL PATHS
# --------------------------
MODEL_PATHS = {
    "XGBoost": "models/xgboost_model.pkl",
    "Random Forest": "models/randomforest_model.pkl",
    "Logistic Regression": "models/logisticregression_model.pkl",
    "TF-IDF Model": "models/tfidf_pipeline.pkl",
    "TF-IDF → RandomForest": "models/tfidf_rf.pkl",
    "TF-IDF → XGBoost": "models/tfidf_xgb.pkl",
    "Char-CNN": "models/charcnn_model.pt",
}

st.set_page_config(page_title="Phishing URL Detector", layout="centered")
st.title("🚨 Phishing URL Detector — Interactive")
st.write("Enter a URL or upload a CSV to classify URLs using multiple trained models.")


# ------------------------------------------------
# SMART MODEL LOADER (supports sklearn + TF-IDF + CharCNN)
# ------------------------------------------------
@st.cache_resource
def load_selected_model(selected_model_name):
    """Loads the chosen model with the correct backend."""
    model_path = MODEL_PATHS[selected_model_name]

    if selected_model_name == "Char-CNN":
        try:
            mdl, vocab, cfg = load_charcnn("models")
            return {"type": "charcnn", "model": mdl, "vocab": vocab, "cfg": cfg}
        except Exception as e:
            st.error(f"❌ Failed to load Char-CNN model: {e}")
            return None

    try:
        mdl = joblib.load(model_path)
        return {"type": "sklearn", "model": mdl}
    except Exception as e:
        st.error(f"❌ Failed to load model {model_path}: {e}")
        return None


# Label mapping helper
def readable_label(label):
    if isinstance(label, str):
        return label.capitalize()
    return "Malicious" if label == 1 else "Benign"


# --------------------------
# SELECT MODEL
# --------------------------
selected_model_name = st.selectbox("Choose model for prediction:", list(MODEL_PATHS.keys()))
model_info = load_selected_model(selected_model_name)
if model_info is None:
    st.stop()


# --------------------------
# SINGLE URL PREDICTION
# --------------------------
st.subheader("🔍 Single URL Prediction")

single_url = st.text_input("Enter a URL", placeholder="e.g. http://example.com/login")

if st.button("Predict Single"):
    if not single_url.strip():
        st.warning("Please enter a URL")
    else:
        try:
            if model_info["type"] == "charcnn":
                pred, prob = predict_single(single_url, model_dir="models")
                st.success(f"Prediction: **{pred.capitalize()}**")
                st.caption(f"P(malicious) = {prob:.4f}")

            else:
                model = model_info["model"]

                if selected_model_name == "TF-IDF Model":
                    pred = model.predict([single_url])[0]
                    st.success(f"Prediction: **{readable_label(pred)}**")

                elif selected_model_name in ["TF-IDF → RandomForest", "TF-IDF → XGBoost"]:
                    mdl = model_info["model"]
                    pred = mdl.predict([single_url])[0]
                    st.success(f"Prediction: **{readable_label(pred)}**")

                    if hasattr(mdl, "predict_proba"):
                        prob = mdl.predict_proba([single_url])[0][1]
                        st.caption(f"P(malicious) = {prob:.4f}")

                else:
                    feats = extract_all_features(single_url)
                    feats_df = pd.DataFrame([feats]).select_dtypes(include=["number", "bool"])

                    pred = model.predict(feats_df)[0]
                    st.success(f"Prediction: **{readable_label(pred)}**")

                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(feats_df)[0][1]
                        st.caption(f"P(malicious) = {prob:.4f}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


st.markdown("---")

# --------------------------
# BATCH PREDICTION (CSV)
# --------------------------
st.subheader("📄 Batch Prediction — Upload CSV")
st.info("CSV must contain a column named `url`.")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)

        # Detect URL column
        url_col = None
        for c in df_in.columns:
            if "url" in c.lower():
                url_col = c
                break

        if url_col is None:
            st.error("❌ No 'url' column found in CSV.")
            st.stop()

        urls = df_in[url_col].astype(str)

        # -----------------------------------------
        # FULL BATCH PREDICTION FOR ALL MODELS
        # -----------------------------------------
        if model_info["type"] == "charcnn":
            mdl = model_info["model"]
            vocab = model_info["vocab"]
            cfg = model_info["cfg"]

            preds, probs = predict_urls(urls.tolist(), mdl, vocab, cfg)

            results = df_in.copy()
            results["prediction"] = ["malicious" if p == 1 else "benign" for p in preds]
            results["malicious_prob"] = probs

        elif selected_model_name == "TF-IDF Model":
            mdl = model_info["model"]
            preds = mdl.predict(urls)

            results = df_in.copy()
            results["prediction"] = [readable_label(p) for p in preds]

        elif selected_model_name in ["TF-IDF → RandomForest", "TF-IDF → XGBoost"]:
            mdl = model_info["model"]
            preds = mdl.predict(urls)

            results = df_in.copy()
            results["prediction"] = [readable_label(p) for p in preds]
            if hasattr(mdl, "predict_proba"):
                results["malicious_prob"] = mdl.predict_proba(urls)[:, 1]

        else:
            mdl = model_info["model"]
            feat_list = [extract_all_features(u) for u in urls]
            feats_df = pd.DataFrame(feat_list).select_dtypes(include=["number", "bool"])

            preds = mdl.predict(feats_df)

            results = df_in.copy()
            results["prediction"] = [readable_label(p) for p in preds]

            if hasattr(mdl, "predict_proba"):
                results["malicious_prob"] = mdl.predict_proba(feats_df)[:, 1]


        # ----------------------------
        # DETECT IF LABELS EXIST → Evaluation Mode
        # ----------------------------
        label_col = None
        for c in ["label", "type", "class", "target", "result"]:
            if c in df_in.columns:
                label_col = c
                break

        if label_col is None:
            # -------------------------------------------
            # PREDICTION MODE (NO LABELS PROVIDED)
            # -------------------------------------------
            st.success("Batch prediction complete!")
            st.dataframe(results.head(200))

            st.info("No label column found → Confusion matrix cannot be generated.")

            st.download_button(
                "Download Predictions CSV",
                results.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

        else:
            # -------------------------------------------
            # EVALUATION MODE (LABELS PROVIDED)
            # -------------------------------------------
            st.success(f"Evaluation Mode: Detected true labels in column '{label_col}'")

            # ---------- CLEAN TRUE LABELS ----------
            def map_label(x):
                x = str(x).strip().lower()
                if x == "benign":
                    return 0
                elif x == "malicious":
                    return 1
                return None

            y_true_num = df_in[label_col].apply(map_label)

            # ---------- CLEAN PREDICTED LABELS ----------
            def map_prediction(x):
                x = str(x).strip().lower()
                if x == "benign":
                    return 0
                elif x == "malicious":
                    return 1
                return None

            y_pred_num = results["prediction"].apply(map_prediction)

            # Confusion Matrix
            cm = confusion_matrix(y_true_num, y_pred_num)

            #st.subheader("📊 Confusion Matrix")
            #st.write(cm)
            cm_df = pd.DataFrame(
                        cm,
                        index=["Actual Benign (0)", "Actual Malicious (1)"],
                        columns=["Pred Benign (0)", "Pred Malicious (1)"]
                    )

            st.subheader("📊 Confusion Matrix")
            st.dataframe(cm_df, use_container_width=True)



            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Classification Report
            st.subheader("📄 Classification Report")
            report = classification_report(y_true_num, y_pred_num, output_dict=True)
            st.json(report)

            acc = accuracy_score(y_true_num, y_pred_num)
            st.metric("Overall Accuracy", f"{acc:.4f}")

            # Show results table
            st.subheader("🔎 Detailed Predictions")
            st.dataframe(results.head(200))

            st.download_button(
                "Download Predictions + Evaluation CSV",
                results.to_csv(index=False).encode("utf-8"),
                file_name="evaluation_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"❌ Failed to process CSV: {e}")


st.markdown("---")

# --------------------------
# MODEL INFO
# --------------------------
st.subheader("ℹ️ Model Information")

if model_info["type"] == "charcnn":
    st.write("Model type: **Char-CNN (PyTorch)**")
else:
    st.write("Model type: **Sklearn**")

st.write(f"Selected model: **{selected_model_name}**")

if model_info["type"] == "sklearn" and hasattr(model_info["model"], "classes_"):
    st.write("Classes:", model_info["model"].classes_)
else:
    st.write("Classes: N/A")

st.caption("This app classifies phishing URLs using classical ML, TF-IDF, and deep learning models.")
