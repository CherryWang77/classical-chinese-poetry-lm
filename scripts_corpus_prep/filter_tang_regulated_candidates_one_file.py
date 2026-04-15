import json
import re
from pathlib import Path
from collections import Counter

INPUT_PATH = Path("全唐诗/poet.song.2000.json")

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

def is_regulated_candidate(paragraphs):
    """
    Prototype rule:
    - split poem into sublines using punctuation
    - keep only poems with exactly 4 or 8 sublines
    - after punctuation removal, all sublines must have same length
    - that length must be 5 or 7
    """
    if not paragraphs:
        return False, None

    sublines = get_poem_sublines(paragraphs)

    if len(sublines) not in {4, 8}:
        return False, None

    lengths = [len(remove_punctuation(s)) for s in sublines]

    if not lengths:
        return False, None

    if len(set(lengths)) != 1:
        return False, None

    if lengths[0] not in {5, 7}:
        return False, None

    info = {
        "num_sublines": len(sublines),
        "subline_length": lengths[0],
        "sublines": sublines,
    }
    return True, info

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    passed_poems = []
    rejection_counts = Counter()

    for poem in data:
        paragraphs = poem.get("paragraphs", [])
        author = poem.get("author")
        title = poem.get("title")
        poem_id = poem.get("id")

        # Detailed rejection tracking
        if not paragraphs:
            rejection_counts["empty_paragraphs"] += 1
            continue

        sublines = get_poem_sublines(paragraphs)

        if len(sublines) not in {4, 8}:
            rejection_counts["not_4_or_8_sublines"] += 1
            continue

        lengths = [len(remove_punctuation(s)) for s in sublines]

        if not lengths:
            rejection_counts["no_lengths"] += 1
            continue

        if len(set(lengths)) != 1:
            rejection_counts["mixed_subline_lengths"] += 1
            continue

        if lengths[0] not in {5, 7}:
            rejection_counts["uniform_but_not_5_or_7"] += 1
            continue

        passed_poems.append({
            "author": author,
            "title": title,
            "id": poem_id,
            "paragraphs": paragraphs,
            "num_sublines": len(sublines),
            "subline_length": lengths[0],
            "sublines": sublines,
        })

    # Summary stats
    by_sublines = Counter(poem["num_sublines"] for poem in passed_poems)
    by_length = Counter(poem["subline_length"] for poem in passed_poems)

    print("\n=== Tang regulated-verse candidate filter (one-file prototype) ===")
    print("Input file:", INPUT_PATH)
    print("Total records in file:", len(data))
    print("Number of candidate poems passing filter:", len(passed_poems))

    print("\nBreakdown by number of sublines:")
    for k, v in sorted(by_sublines.items()):
        print(f"  {k} sublines -> {v} poems")

    print("\nBreakdown by subline length:")
    for k, v in sorted(by_length.items()):
        print(f"  length {k} -> {v} poems")

    print("\nRejection reasons:")
    for reason, count in rejection_counts.most_common():
        print(f"  {reason}: {count}")

    print("\nFirst 5 passing examples:")
    for i, poem in enumerate(passed_poems[:5], start=1):
        print(f"\nExample {i}")
        print("Author:", poem["author"])
        print("Title:", poem["title"])
        print("ID:", poem["id"])
        print("num_sublines:", poem["num_sublines"])
        print("subline_length:", poem["subline_length"])
        print("Original paragraphs:")
        for line in poem["paragraphs"]:
            print(" ", line)
        print("Split sublines:")
        for s in poem["sublines"]:
            print(" ", s)

if __name__ == "__main__":
    main()