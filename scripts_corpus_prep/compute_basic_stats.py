from pathlib import Path

def compute_stats(file_path: Path):
    # Read whole file
    text = file_path.read_text(encoding="utf-8")

    # File size in bytes
    file_size = file_path.stat().st_size

    # Poems are separated by blank lines
    poems = [p for p in text.strip().split("\n\n") if p.strip()]

    # Split each poem into lines
    poem_lines = [poem.split("\n") for poem in poems]

    # Flatten all lines
    all_lines = [line for poem in poem_lines for line in poem if line.strip()]

    # Character counts per line
    line_char_counts = [len(line) for line in all_lines]

    # Basic stats
    num_poems = len(poems)
    num_lines = len(all_lines)
    total_characters = sum(line_char_counts)

    avg_lines_per_poem = num_lines / num_poems if num_poems else 0
    avg_chars_per_line = total_characters / num_lines if num_lines else 0

    print(f"\n=== Stats for: {file_path} ===")
    print("File size (bytes):", file_size)
    print("Number of poems:", num_poems)
    print("Number of lines:", num_lines)
    print("Total characters:", total_characters)
    print("Average lines per poem:", round(avg_lines_per_poem, 2))
    print("Average characters per line:", round(avg_chars_per_line, 2))


# Change these paths if needed
songci_file = Path("songci_1000_export.txt")
tang_file = Path("tang_2000_export.txt")

compute_stats(songci_file)
compute_stats(tang_file)