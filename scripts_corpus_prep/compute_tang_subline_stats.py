from pathlib import Path
from collections import Counter
import re

# We will mainly split Tang lines at major Chinese punctuation marks
SPLIT_PUNCT = "，。！？；：、"
REMOVE_PUNCT = SPLIT_PUNCT + "（）《》〈〉“”‘’〔〕【】—…·,.!?;:()[]\"' "

def remove_punctuation(text: str) -> str:
    return "".join(ch for ch in text if ch not in REMOVE_PUNCT)

def read_poems(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    poems = [p for p in text.strip().split("\n\n") if p.strip()]
    poem_lines = []
    for poem in poems:
        lines = [line for line in poem.split("\n") if line.strip()]
        poem_lines.append(lines)
    return poem_lines

def split_line_into_subunits(line: str):
    """
    Split one exported Tang line into smaller units using major punctuation.
    Empty pieces are removed.
    """
    pattern = "[" + re.escape(SPLIT_PUNCT) + "]"
    parts = re.split(pattern, line)
    return [part.strip() for part in parts if part.strip()]

def top_n(counter_obj, n=15):
    return counter_obj.most_common(n)

def compute_tang_subline_stats(file_path: Path):
    poems = read_poems(file_path)

    subline_length_distribution = Counter()
    subline_count_per_poem_distribution = Counter()

    total_sublines = 0

    print(f"\n=== Tang subline stats for: {file_path} ===")

    for poem in poems:
        sublines_in_this_poem = []

        for line in poem:
            parts = split_line_into_subunits(line)
            sublines_in_this_poem.extend(parts)

            for part in parts:
                stripped = remove_punctuation(part)
                subline_length_distribution[len(stripped)] += 1
                total_sublines += 1

        subline_count_per_poem_distribution[len(sublines_in_this_poem)] += 1

    print("Number of poems:", len(poems))
    print("Total sublines after punctuation splitting:", total_sublines)

    print("\nTop subline counts per poem:")
    for count, freq in top_n(subline_count_per_poem_distribution, 15):
        print(f"  {count} sublines -> {freq} poems")

    print("\nTop subline lengths after punctuation removal:")
    for length, freq in top_n(subline_length_distribution, 20):
        print(f"  length {length} -> {freq} sublines")

    print("\nExamples: first 5 poems with line splitting")
    for i, poem in enumerate(poems[:5], start=1):
        print(f"\nPoem {i}:")
        for line in poem:
            parts = split_line_into_subunits(line)
            print(f"  Original line: {line}")
            for part in parts:
                stripped = remove_punctuation(part)
                print(f"    part: {part}   [len_no_punct={len(stripped)}]")

tang_file = Path("tang_2000_export.txt")
compute_tang_subline_stats(tang_file)