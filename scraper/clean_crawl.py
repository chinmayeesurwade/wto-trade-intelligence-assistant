import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import os
import time

os.makedirs("data/cleaned", exist_ok=True)

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

        # Remove junk tags
        for tag in soup([
            "script",
            "style",
            "nav",
            "header",
            "footer"
        ]):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else ""

        text = soup.get_text("\n", strip=True)

        lines = []

        for line in text.split("\n"):

            line = line.strip()

            if len(line) < 30:
                continue

            lines.append(line)

        clean_text = "\n".join(lines)

        data = {
            "url": url,
            "title": title,
            "content": clean_text
        }

        with open(
            f"data/cleaned/page_{idx}.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        time.sleep(1)

    except Exception as e:
        print("ERROR:", e)

print("Done")