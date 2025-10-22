# test_features.py
"""
Unit tests for feature extraction.
"""

from src.features import extract_all_features

def test_url_features_basic():
    url = "https://tinyurl.com/abcd"
    feats = extract_all_features(url)
    assert "entropy" in feats
    assert "is_shortener" in feats
    assert feats["is_shortener"] == 1

def test_html_features_counts():
    html = "<html><body><form></form><script></script><iframe></iframe></body></html>"
    feats = extract_all_features("https://example.com", html)
    assert feats["num_forms"] == 1
    assert feats["num_iframes"] == 1
    assert feats["num_scripts"] == 1
