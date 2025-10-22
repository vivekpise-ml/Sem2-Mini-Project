# test_models.py
"""
Basic model training test.
"""

import os
import pandas as pd
from src.train_classical import train_models
from src.config import MODEL_DIR

def test_model_training(tmp_path):
    # Create dummy dataset
    df = pd.DataFrame({
        "url": ["https://google.com", "http://secure-login-bank.com/verify"],
        "label": [0, 1]
    })
    
    results = train_models(df)
    for name, r in results.items():
        assert "accuracy" in r
        assert os.path.exists(r["model_path"])
