# features.py
"""
Feature extraction module for phishing detection.
Includes:
- URL-based features (length, entropy, keyword checks, etc.)
- HTML-based features (forms, scripts, iframes, event handlers)
"""

import re
import math
from collections import Counter
from bs4 import BeautifulSoup
import tldextract

# -------------------------------------------------------------------
# Known URL shorteners
# -------------------------------------------------------------------
SHORTENER_DOMAINS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "buff.ly", "is.gd", "rebrand.ly", "lnkd.in", "shorturl.at"
]

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def shannon_entropy(s: str) -> float:
    """Compute Shannon entropy (measure of randomness) of a string."""
    if not s:
        return 0.0
    counts = Counter(s)
    probs = [c / len(s) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)

def is_shortener(url: str) -> int:
    """Check if URL belongs to known shortening services."""
    return int(any(dom in url.lower() for dom in SHORTENER_DOMAINS))

# -------------------------------------------------------------------
# URL feature extraction
# -------------------------------------------------------------------
def extract_url_features(url: str) -> dict:
    """Extract numerical and categorical features from a given URL."""
    features = {}

    # Basic structure
    features["url_len"] = len(url)
    features["count_dots"] = url.count(".")
    features["count_hyphen"] = url.count("-")
    features["count_at"] = url.count("@")
    features["count_qm"] = url.count("?")
    features["count_percent"] = url.count("%")
    features["num_digits"] = sum(c.isdigit() for c in url)

    # Pattern-based
    features["has_ip"] = int(bool(re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", url)))

    # Entropy
    features["entropy"] = shannon_entropy(url)

    # Suspicious tokens
    suspicious_tokens = ["login", "secure", "verify", "update", "confirm", "account", "bank", "ebay", "paypal"]
    features["suspicious_token"] = int(any(tok in url.lower() for tok in suspicious_tokens))

    # Domain extraction
    domain_info = tldextract.extract(url)
    domain = f"{domain_info.domain}.{domain_info.suffix}" if domain_info.suffix else domain_info.domain
    features["domain"] = domain
    features["is_shortener"] = is_shortener(domain)

    return features

# -------------------------------------------------------------------
# HTML feature extraction
# -------------------------------------------------------------------
def extract_html_features(html: str) -> dict:
    """Extract features from HTML content."""
    soup = BeautifulSoup(html or "<html></html>", "html.parser")
    features = {}

    features["num_forms"] = len(soup.find_all("form"))
    features["num_iframes"] = len(soup.find_all("iframe"))
    features["num_scripts"] = len(soup.find_all("script"))
    features["num_links"] = len(soup.find_all("a"))
    features["has_event_handlers"] = int(
        bool(soup.find_all(attrs={"onload": True})) or bool(soup.find_all(attrs={"onclick": True}))
    )

    return features

# -------------------------------------------------------------------
# Combined feature extraction
# -------------------------------------------------------------------
def extract_all_features(url: str, html: str = None) -> dict:
    """Combine both URL and HTML-based features into a single dictionary."""
    features = extract_url_features(url)
    if html:
        features.update(extract_html_features(html))
    else:
        # If HTML is missing, fill HTML features with zeros
        features.update({
            "num_forms": 0,
            "num_iframes": 0,
            "num_scripts": 0,
            "num_links": 0,
            "has_event_handlers": 0,
        })
    return features

# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    sample_url = "https://tinyurl.com/2p9fb9kw"
    sample_html = "<html><body><form></form><script></script></body></html>"

    feats = extract_all_features(sample_url, sample_html)
    for k, v in feats.items():
        print(f"{k}: {v}")
