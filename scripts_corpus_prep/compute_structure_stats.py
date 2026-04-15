from pathlib import Path
from collections import Counter

# Common punctuation to remove for simple line-length counting
PUNCT_TO_REMOVE = "，。！？；：、（）《》〈〉“”‘’〔〕【】—…·,.!?;:()[]\"' "

def remove_punctuation(text: str) -> str:
    return "".join(ch for ch in text if ch not in PUNCT_TO_REMOVE)

def read_poems(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    poems = [p for p in text.strip().split("\n\n") if p.strip()]
    poem_lines = []
    for poem in poems:
        lines = [line for line in poem.split("\n") if line.strip()]
        poem_lines.append(lines)
    return poem_lines

def top_n(counter_obj, n=10):
    return counter_obj.most_common(n)

def compute_structure_stats(file_path: Path):
    poems = read_poems(file_path)

    # 1) number of lines per poem
    lines_per_poem = [len(poem) for poem in poems]
    line_count_distribution = Counter(lines_per_poem)

    # 2) line lengths
    raw_line_lengths = []
    stripped_line_lengths = []

    for poem in poems:
        for line in poem:
            raw_line_lengths.append(len(line))
            stripped = remove_punctuation(line)
            stripped_line_lengths.append(len(stripped))

    raw_line_length_distribution = Counter(raw_line_lengths)
    stripped_line_length_distribution = Counter(stripped_line_lengths)

    print(f"\n=== Structure stats for: {file_path} ===")
    print("Number of poems:", len(poems))

    print("\nTop line-counts per poem (most common):")
    for count, freq in top_n(line_count_distribution, 10):
        print(f"  {count} lines -> {freq} poems")

    print("\nTop raw line lengths (including punctuation):")
    for length, freq in top_n(raw_line_length_distribution, 15):
        print(f"  length {length} -> {freq} lines")

    print("\nTop line lengths after punctuation removal:")
    for length, freq in top_n(stripped_line_length_distribution, 15):
        print(f"  length {length} -> {freq} lines")

    print("\nExamples: first 3 poems with stripped line lengths")
    for i, poem in enumerate(poems[:3], start=1):
        print(f"\nPoem {i}:")
        for line in poem:
            stripped = remove_punctuation(line)
            print(f"  {line}   [len_no_punct={len(stripped)}]")

songci_file = Path("songci_1000_export.txt")
tang_file = Path("tang_2000_export.txt")

compute_structure_stats(songci_file)
compute_structure_stats(tang_file)