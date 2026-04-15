from pathlib import Path

# Existing exported corpus files
SONGCI_INPUT = Path("songci_multi_file_export.txt")
TANG_INPUT = Path("tang_regulated_candidates_multi_file.txt")

# Target sizes in bytes (approximate)
ONE_MB = 1024 * 1024

TARGETS = {
    "pilot_1_5mb": int(1.5 * ONE_MB),
    "main_5mb": int(5 * ONE_MB),
    "extended_10mb": int(10 * ONE_MB),
}

def read_poems_from_txt(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    poems = [p for p in text.strip().split("\n\n") if p.strip()]
    return poems

def build_subset_by_target_size(poems, target_bytes):
    """
    Add whole poems one by one until we reach or slightly exceed target size.
    We preserve poem boundaries by never cutting inside a poem.
    """
    selected = []
    current_text = ""
    current_size = 0

    for poem in poems:
        if not selected:
            candidate_text = poem
        else:
            candidate_text = current_text + "\n\n" + poem

        candidate_size = len(candidate_text.encode("utf-8"))

        if candidate_size <= target_bytes:
            selected.append(poem)
            current_text = candidate_text
            current_size = candidate_size
        else:
            # Decide whether adding this poem gets us closer than stopping here
            diff_without = abs(target_bytes - current_size)
            diff_with = abs(target_bytes - candidate_size)

            if diff_with < diff_without:
                selected.append(poem)
                current_text = candidate_text
                current_size = candidate_size
            break

    return selected, current_text, current_size

def write_subset(output_path: Path, text: str):
    output_path.write_text(text, encoding="utf-8")

def make_subsets_for_corpus(input_path: Path, prefix: str):
    poems = read_poems_from_txt(input_path)

    print(f"\n=== Building subsets for: {input_path} ===")
    print("Number of poems available:", len(poems))
    print("Input size (bytes):", input_path.stat().st_size)

    for label, target_bytes in TARGETS.items():
        selected_poems, subset_text, subset_size = build_subset_by_target_size(poems, target_bytes)

        output_path = Path(f"{prefix}_{label}.txt")
        write_subset(output_path, subset_text)

        print(f"\nSubset: {label}")
        print("  target bytes:", target_bytes)
        print("  actual bytes:", subset_size)
        print("  poems included:", len(selected_poems))
        print("  output file:", output_path)

def main():
    make_subsets_for_corpus(SONGCI_INPUT, "songci")
    make_subsets_for_corpus(TANG_INPUT, "tang")

if __name__ == "__main__":
    main()