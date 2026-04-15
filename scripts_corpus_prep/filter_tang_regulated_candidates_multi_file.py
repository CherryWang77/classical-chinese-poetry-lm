import json
import re
from pathlib import Path
from collections import Counter

# Folder containing the Tang-side files
INPUT_DIR = Path("全唐诗")

# We only match the file pattern that you have already observed
FILE_PATTERN = "poet.song.*.json"

# Output files
OUTPUT_JSON = Path("tang_regulated_candidates_multi_file.json")
OUTPUT_TXT = Path("tang_regulated_candidates_multi_file.txt")

SPLIT_PUNCT = "，。！？；：、"
REMOVE_PUNCT = SPLIT_PUNCT + "（）《》〈〉“”‘’〔〕【】—…·,.!?;:()[]\"' "

def remove_punctuation(text: str) -> str:
    return "".join(ch for ch in text if ch not in REMOVE_PUNCT)

def split_line_into_subunits(line: str):
    pattern = "[" + re.escape(SPLIT_PUNCT) + "]"
    parts = re.split(pattern, line)
    return [part.strip() for part in parts if part.strip()]

def get_poem_sublines(paragraphs):
    sublines = []
    for line in paragraphs:
        parts = split_line_into_subunits(line)
        sublines.extend(parts)
    return sublines

def classify_candidate(paragraphs):
    """
    Prototype candidate rule:
    - split poem into sublines using punctuation
    - require exactly 4 or 8 sublines
    - after punctuation removal, all sublines must have the same length
    - that length must be 5 or 7
    """
    if not paragraphs:
        return False, "empty_paragraphs", None

    sublines = get_poem_sublines(paragraphs)

    if len(sublines) not in {4, 8}:
        return False, "not_4_or_8_sublines", None

    lengths = [len(remove_punctuation(s)) for s in sublines]

    if not lengths:
        return False, "no_lengths", None

    if len(set(lengths)) != 1:
        return False, "mixed_subline_lengths", None

    if lengths[0] not in {5, 7}:
        return False, "uniform_but_not_5_or_7", None

    info = {
        "num_sublines": len(sublines),
        "subline_length": lengths[0],
        "sublines": sublines,
    }
    return True, "passed", info

def poem_to_text(poem_record):
    paragraphs = poem_record.get("paragraphs", [])
    return "\n".join(paragraphs)

def main():
    input_files = sorted(INPUT_DIR.glob(FILE_PATTERN))

    print("Matched files:", len(input_files))
    if not input_files:
        print("No files matched. Check INPUT_DIR and FILE_PATTERN.")
        return

    all_passed_poems = []

    total_records = 0
    file_level_summary = []
    global_rejection_counts = Counter()

    for file_path in input_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_total = len(data)
        file_passed = 0
        file_rejections = Counter()

        total_records += file_total

        for poem in data:
            paragraphs = poem.get("paragraphs", [])
            author = poem.get("author")
            title = poem.get("title")
            poem_id = poem.get("id")

            passed, reason, info = classify_candidate(paragraphs)

            if not passed:
                file_rejections[reason] += 1
                global_rejection_counts[reason] += 1
                continue

            file_passed += 1

            all_passed_poems.append({
                "source_file": str(file_path),
                "author": author,
                "title": title,
                "id": poem_id,
                "paragraphs": paragraphs,
                "num_sublines": info["num_sublines"],
                "subline_length": info["subline_length"],
                "sublines": info["sublines"],
            })

        file_level_summary.append({
            "file": str(file_path),
            "total_records": file_total,
            "passed": file_passed,
            "rejections": dict(file_rejections),
        })

    # Global summaries
    by_sublines = Counter(poem["num_sublines"] for poem in all_passed_poems)
    by_length = Counter(poem["subline_length"] for poem in all_passed_poems)

    # Export JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_passed_poems, f, ensure_ascii=False, indent=2)

    # Export plain text
    text_poems = [poem_to_text(poem) for poem in all_passed_poems if poem.get("paragraphs")]
    final_text = "\n\n".join(text_poems)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(final_text)

    # Print report
    print("\n=== Tang regulated-verse candidate filter (multi-file prototype) ===")
    print("Input dir:", INPUT_DIR)
    print("File pattern:", FILE_PATTERN)
    print("Matched files:", len(input_files))
    print("Total records scanned:", total_records)
    print("Total candidate poems passing filter:", len(all_passed_poems))

    print("\nBreakdown by number of sublines:")
    for k, v in sorted(by_sublines.items()):
        print(f"  {k} sublines -> {v} poems")

    print("\nBreakdown by subline length:")
    for k, v in sorted(by_length.items()):
        print(f"  length {k} -> {v} poems")

    print("\nGlobal rejection reasons:")
    for reason, count in global_rejection_counts.most_common():
        print(f"  {reason}: {count}")

    print("\nOutput files:")
    print("  JSON:", OUTPUT_JSON, "| bytes =", OUTPUT_JSON.stat().st_size)
    print("  TXT :", OUTPUT_TXT, "| bytes =", OUTPUT_TXT.stat().st_size)

    print("\nFirst 10 file summaries:")
    for item in file_level_summary[:10]:
        print(f"\nFile: {item['file']}")
        print("  total_records:", item["total_records"])
        print("  passed:", item["passed"])
        print("  rejections:", item["rejections"])

    print("\nFirst 5 passing examples:")
    for i, poem in enumerate(all_passed_poems[:5], start=1):
        print(f"\nExample {i}")
        print("Source file:", poem["source_file"])
        print("Author:", poem["author"])
        print("Title:", poem["title"])
        print("ID:", poem["id"])
        print("num_sublines:", poem["num_sublines"])
        print("subline_length:", poem["subline_length"])
        print("Original paragraphs:")
        for line in poem["paragraphs"]:
            print(" ", line)

if __name__ == "__main__":
    main()