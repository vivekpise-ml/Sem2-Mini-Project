"""
Main entrypoint for the Phishing URL Detection project.

This script:
1. Loads the dataset
2. Trains multiple classical ML models (RF, LR, XGBoost)
3. Trains the RNN model on raw URL sequences ---- This is for the next phase
4. Saves all trained models in the /models directory
"""

import pandas as pd
#from src import train_models, train_rnn_model
from src.train_classical import train_models
from src.config import DATA_PATH
from src.evaluate import evaluate_models

def main():
    print("=" * 80)
    print("🚀 PHISHING URL DETECTION PROJECT - TRAINING PIPELINE")
    print("=" * 80)
    
    # --- Phase 1: Load Dataset ---
    print("\n📥 Loading dataset...")
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Dataset loaded successfully! Total samples: {len(df)}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {DATA_PATH}. Please check the path in config.py.")
        return

    print("\n📋 Columns in dataset:", df.columns.tolist())
    print(df.head())
    # --- Phase 2: Train Classical ML Models ---
    print("\n🧠 Training Classical ML Models (RandomForest, LogisticRegression, XGBoost)...")
    model_results = train_models(df)
    print("\n📊 Model Performance Summary:")
    for model_name, metrics in model_results.items():
        print(f"   {model_name:<25} Accuracy: {metrics['accuracy']:.4f}")

    '''
    # --- Phase 3: Train RNN Model ---
    print("\n🤖 Training RNN (LSTM/GRU) on URL text sequences...")
    try:
        rnn_accuracy = train_rnn_model(df)
        print(f"✅ RNN Model Accuracy: {rnn_accuracy:.4f}")
    except Exception as e:
        print(f"⚠️ RNN training skipped or failed: {e}")
    '''

    print("\n🏁 Training complete! All models are saved under the /models directory.")
    print("=" * 80)

    # ... after training
    print("\n📊 Evaluating models after training...")
    evaluate_models(df)


if __name__ == "__main__":
    main()
