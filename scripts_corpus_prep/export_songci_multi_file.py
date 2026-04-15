import json
from pathlib import Path

# Folder containing Song ci files
INPUT_DIR = Path("宋词")

# Prototype file pattern based on the file you already inspected: ci.song.1000.json
FILE_PATTERN = "ci.song.*.json"

# Output text file
OUTPUT_TXT = Path("songci_multi_file_export.txt")

def poem_to_text(poem_record):
    paragraphs = poem_record.get("paragraphs", [])
    if not paragraphs:
        return None
    return "\n".join(paragraphs)

def main():
    input_files = sorted(INPUT_DIR.glob(FILE_PATTERN))

    print("Matched files:", len(input_files))
    if not input_files:
        print("No files matched. Check INPUT_DIR and FILE_PATTERN.")
        return

    exported_poems = []
    total_records = 0
    skipped_empty = 0
    file_level_summary = []

    for file_path in input_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_total = len(data)
        file_exported = 0
        file_skipped_empty = 0

        total_records += file_total

        for poem in data:
            poem_text = poem_to_text(poem)
            if poem_text is None:
                skipped_empty += 1
                file_skipped_empty += 1
                continue

            exported_poems.append(poem_text)
            file_exported += 1

        file_level_summary.append({
            "file": str(file_path),
            "total_records": file_total,
            "exported": file_exported,
            "skipped_empty": file_skipped_empty,
        })

    final_text = "\n\n".join(exported_poems)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(final_text)

    print("\n=== Song ci multi-file export ===")
    print("Input dir:", INPUT_DIR)
    print("File pattern:", FILE_PATTERN)
    print("Matched files:", len(input_files))
    print("Total records scanned:", total_records)
    print("Total poems exported:", len(exported_poems))
    print("Skipped empty poems:", skipped_empty)
    print("Output file:", OUTPUT_TXT)
    print("Output size in bytes:", OUTPUT_TXT.stat().st_size)

    print("\nFirst 10 file summaries:")
    for item in file_level_summary[:10]:
        print(f"\nFile: {item['file']}")
        print("  total_records:", item["total_records"])
        print("  exported:", item["exported"])
        print("  skipped_empty:", item["skipped_empty"])

    print("\nFirst 3 exported poems preview:")
    for i, poem_text in enumerate(exported_poems[:3], start=1):
        print(f"\nPoem {i}:")
        print(poem_text)

if __name__ == "__main__":
    main()