import os
import json

found = 0

for file in os.listdir("data/cleaned"):

    if not file.endswith(".json"):
        continue

    path = os.path.join("data/cleaned", file)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    text = data["content"]

    if "trade for peace" in text.lower():

        found += 1

        print("\nFOUND IN:", file)
        print("=" * 60)

        idx = text.lower().find("trade for peace")

        start = max(0, idx - 300)
        end = min(len(text), idx + 1200)

        print(text[start:end])

print("\nTOTAL FOUND:", found)