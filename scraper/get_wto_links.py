import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://www.wto.org/english/info_e/site2_e.htm"

response = requests.get(URL, timeout=30)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

links = []

for a in soup.find_all("a", href=True):
    href = a["href"]

    if href.startswith("/"):
        href = "https://www.wto.org" + href

    if "wto.org" in href:
        links.append(href)

links = list(set(links))

df = pd.DataFrame({"url": links})

df.to_csv("data/wto_urls.csv", index=False)

print(f"Collected {len(links)} URLs")