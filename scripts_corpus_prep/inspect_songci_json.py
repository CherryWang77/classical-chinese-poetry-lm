import json
from pathlib import Path

file_path = Path("宋词/ci.song.1000.json")

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Top-level type:", type(data))
print("Number of records in this file:", len(data))

first = data[0]

print("\nKeys in first record:")
print(first.keys())

print("\nAuthor:")
print(first.get("author"))

print("\nRhythmic:")
print(first.get("rhythmic"))

print("\nParagraphs:")
print(first.get("paragraphs"))

print("\nPoem as lines:")
for i, line in enumerate(first.get("paragraphs", []), start=1):
    print(f"{i}: {line}")