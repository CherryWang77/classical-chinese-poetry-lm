import json
from pathlib import Path

# 1. change this path to the file you want to inspect
file_path = Path("全唐诗/poet.song.2000.json")

# 2. read json
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. basic checks
print("Top-level type:", type(data))
print("Number of records in this file:", len(data))

# 4. inspect the first record
first = data[0]

print("\nKeys in first record:")
print(first.keys())

print("\nAuthor:")
print(first.get("author"))

print("\nTitle:")
print(first.get("title"))

print("\nID:")
print(first.get("id"))

print("\nParagraphs:")
print(first.get("paragraphs"))

print("\nPoem as lines:")
for i, line in enumerate(first.get("paragraphs", []), start=1):
    print(f"{i}: {line}")
