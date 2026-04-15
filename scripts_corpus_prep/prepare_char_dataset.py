from pathlib import Path
import json

# Change this if you want to test another corpus file later
INPUT_FILE = Path("songci_pilot_1_5mb.txt")

# Output files for vocabulary and encoded data preview
VOCAB_JSON = Path("songci_pilot_vocab.json")
ENCODED_PREVIEW_JSON = Path("songci_pilot_encoded_preview.json")

# Validation split ratio
VAL_RATIO = 0.1

def main():
    # 1. Read raw text
    text = INPUT_FILE.read_text(encoding="utf-8")

    print("=== Character-level dataset preparation ===")
    print("Input file:", INPUT_FILE)
    print("Input size in bytes:", INPUT_FILE.stat().st_size)
    print("Total characters in corpus:", len(text))

    # 2. Build vocabulary
    vocab = sorted(set(text))
    vocab_size = len(vocab)

    print("Unique characters (vocab size):", vocab_size)

    # 3. Build stoi / itos
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    # 4. Encode full text as integer IDs
    encoded = [stoi[ch] for ch in text]

    print("Encoded sequence length:", len(encoded))

    # 5. Train / validation split
    split_idx = int(len(encoded) * (1 - VAL_RATIO))
    train_ids = encoded[:split_idx]
    val_ids = encoded[split_idx:]

    print("Train length:", len(train_ids))
    print("Validation length:", len(val_ids))
    print("Validation ratio:", VAL_RATIO)

    # 6. Save vocabulary
    vocab_payload = {
        "input_file": str(INPUT_FILE),
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": {str(i): ch for i, ch in itos.items()},
    }

    with open(VOCAB_JSON, "w", encoding="utf-8") as f:
        json.dump(vocab_payload, f, ensure_ascii=False, indent=2)

    # 7. Save a small preview of encoded data
    preview_payload = {
        "input_file": str(INPUT_FILE),
        "first_200_chars": text[:200],
        "first_200_encoded_ids": encoded[:200],
        "train_preview_ids": train_ids[:100],
        "val_preview_ids": val_ids[:100],
    }

    with open(ENCODED_PREVIEW_JSON, "w", encoding="utf-8") as f:
        json.dump(preview_payload, f, ensure_ascii=False, indent=2)

    # 8. Print a small readable preview
    print("\nFirst 200 raw characters:")
    print(text[:200])

    print("\nFirst 50 encoded IDs:")
    print(encoded[:50])

    print("\nOutput files:")
    print("  Vocab JSON:", VOCAB_JSON)
    print("  Encoded preview JSON:", ENCODED_PREVIEW_JSON)

if __name__ == "__main__":
    main()