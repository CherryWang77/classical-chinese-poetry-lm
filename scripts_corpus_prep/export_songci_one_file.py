import json
from pathlib import Path

# Input JSON file
input_path = Path("宋词/ci.song.1000.json")

# Output text file
output_path = Path("songci_1000_export.txt")

# Read JSON
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Collect exported poems
exported_poems = []

for poem in data:
    paragraphs = poem.get("paragraphs", [])

    # Keep only poems that have at least one line
    if not paragraphs:
        continue

    # Join poem lines with newline
    poem_text = "\n".join(paragraphs)

    # Add to export list
    exported_poems.append(poem_text)

# Join poems with a blank line between them
final_text = "\n\n".join(exported_poems)

# Write output
with open(output_path, "w", encoding="utf-8") as f:
    f.write(final_text)

print("Export finished.")
print("Input file:", input_path)
print("Output file:", output_path)
print("Number of exported poems:", len(exported_poems))
print("Output size in bytes:", output_path.stat().st_size)