import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import os
import time

os.makedirs("data/raw", exist_ok=True)

urls = pd.read_csv("data/wto_urls.csv")

for idx, row in urls.iterrows():

    url = row["url"]

    try:
        print(f"[{idx+1}/{len(urls)}] {url}")

        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.text.strip() if soup.title else ""

        text = soup.get_text(" ", strip=True)

        data = {
            "url": url,
            "title": title,
            "content": text
        }

        filename = f"data/raw/page_{idx}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        time.sleep(1)

    except Exception as e:
        print("ERROR:", e)

print("Done")