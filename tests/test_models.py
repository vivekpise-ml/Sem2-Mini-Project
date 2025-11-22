# test_models.py
"""
Basic model training test.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import os
import pandas as pd
from src.train_classical import train_models
from src.config import MODEL_DIR

def test_model_training(tmp_path):
    # Create dummy dataset
    '''
    df = pd.DataFrame({
        "url": ["https://google.com", "http://secure-login-bank.com/verify"],
        "label": [0, 1]
    })
    '''

    df = pd.DataFrame({
    "url": [
        "https://google.com",
        "https://example.com",
        "https://openai.com",
        "http://secure-login-bank.com/verify",
        "http://phishy-login.net/update",
        "http://malicious-login-site.org",
        ],
        "label": [0, 0, 0, 1, 1, 1]
    })


    
    results = train_models(df)
    for name, r in results.items():
        assert "accuracy" in r
        assert os.path.exists(r["model_path"])
