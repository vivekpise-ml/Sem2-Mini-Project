Malicious URL Detection Using ML & Deep Learning

Malicious URL Detection

A Machine Learning + Deep Learning based system to classify URLs as Benign or Malicious using Classical ML, TF-IDF Models, and Character-Level CNN.

Project Overview

Phishing and malicious URLs remain a major cybersecurity threat.
This project builds an end-to-end ML pipeline to detect such URLs using:

Classical ML Models (LR, RF, XGB)

TF-IDF + ML models

Character-Level CNN (CharCNN)

DVC for dataset versioning

Streamlit for deployment

The system extracts handcrafted URL features, converts URLs into text vectors using TF-IDF, trains multiple classifiers, and exposes a real-time prediction interface using Streamlit.

                ┌──────────────────┐
                │    Raw URL        │
                └───────┬──────────┘
                        │
                ┌───────▼────────────┐
                │  Feature Extraction │
                │ (src/extract_all..)│
                └───────┬────────────┘
                        │
   ┌──────────────┬─────┼───────┬────────────────┬──────────────────┐
   │ Classical ML │ TF-IDF+ML   │ TF-IDF+XGBoost │   Char-CNN       │
   └──────────────┴─────┼───────┴────────────────┴──────────────────┘
                        │
                ┌───────▼────────────┐
                │   Prediction        │
                └───────┬────────────┘
                        │
                ┌───────▼────────────┐
                │   Streamlit UI      │
                └─────────────────────┘

Features Extracted

Hand-crafted numerical features include:

URL length

Number of digits

Number of special symbols

Number of suspicious keywords (e.g., login, secure, verify)

Entropy

Number of subdomains

TLD length

Presence of IP addresses

Domain age (if available)

Symbol ratios (e.g., %, @, =)


Models Trained
1️⃣ Classical Machine Learning

Logistic Regression

Random Forest

XGBoost (numeric features only)

2️⃣ TF-IDF Based Models

TF-IDF + Logistic Regression

TF-IDF + Random Forest

TF-IDF + XGBoost

3️⃣ Deep Learning

Character-Level Convolutional Neural Network (CharCNN)

Model Performance Summary
Model	Accuracy	Notes
Random Forest (numeric)	~0.95	Best classical model
TF-IDF + XGBoost	~0.95	Highest among text models
CharCNN	~0.88	Learns raw character sequences

Exact metrics are available in the metrics/ directory.


Sem2-Mini-Project
│
├── app/                     # Streamlit application
│   └── streamlit_app.py
│
├── src/                     # Feature extraction & training
│   ├── extract_all_features.py
│   ├── train_classical.py
│   ├── train_tfidf.py
│   ├── train_charcnn.py
│   └── evaluate.py
|   |___ visualize.py
│
├── models/                  # Pre-trained models (ignored in Git)
│   ├── randomforest_model.pkl
│   ├── tfidf_rf.pkl
│   ├── xgboost_model.pkl
│   ├── charcnn_model.pt
│   └── scalers, vectorizers
│
├── data/
│   └── kaggle/
│       └── balanced_urls.csv.dvc   # Tracked by DVC (dataset not in Git)
│
├── metrics/                 # Confusion matrices, accuracy, heatmaps
│
├── .dvc/                    # DVC metadata
├── dvc_store/               # Local DVC remote
│
├── README.md
└── requirements.txt
|___ main.py
|


Dataset Versioning with DVC

Large datasets are not pushed to GitHub.

Dataset used:
data/kaggle/balanced_urls.csv

dvc add data/kaggle/balanced_urls.csv
git add data/kaggle/balanced_urls.csv.dvc
git commit -m "Track dataset with DVC"
dvc push

To download dataset later:

dvc pull


Running the Streamlit App

streamlit run app/streamlit_app.py


What the app provides:

Input box for URL

Real-time prediction

Probability distribution

Extracted features

Visual explanation

Training the Models

All training scripts live in src/.

Examples:

Classical ML:
python src/train_classical.py

TF-IDF models:
python src/train_tfidf.py

CharCNN:
python src/train_charcnn.py

Regenerating Plots

The plots/ directory is ignored by Git.

To regenerate visualizations:

python src/visualize.py


This recreates:

Confusion matrices

ROC curves

F1-score bar charts

TF-IDF model comparisons


Deployment Options

Local Streamlit

Docker container (optional)

FastAPI endpoint (optional upgrade)
