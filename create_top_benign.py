import pandas as pd
import requests

TRANCE_SOURCE = "https://raw.githubusercontent.com/saffranxu/data/main/tranco_1m.csv"

def download_tranco_from_github():
    print("Downloading Tranco top 1M domains from GitHub…")

    r = requests.get(TRANCE_SOURCE)
    text = r.text.strip()

    rows = text.split("\n")

    domains = []
    for row in rows:
        parts = row.split(",")
        if len(parts) == 2:
            domains.append(parts[1].strip())

    print("✔ Downloaded", len(domains), "domains from GitHub mirror.")
    return domains


def expand_domains_to_urls(domains):
    urls = []
    for d in domains:
        urls.append(f"http://{d}")
        urls.append(f"https://{d}")
        urls.append(f"https://www.{d}")
    return urls


if __name__ == "__main__":
    domains = download_tranco_from_github()

    # take top 8000 → about 24,000 URLs
    domains = domains[:8000]

    urls = expand_domains_to_urls(domains)

    df = pd.DataFrame({"url": urls})
    df["label"] = "benign"

    df.to_csv("benign_real.csv", index=False)
    print("✔ Saved benign_real.csv with", len(df), "rows.")
