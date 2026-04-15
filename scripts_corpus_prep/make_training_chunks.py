from pathlib import Path

# Input corpus
INPUT_FILE = Path("songci_pilot_1_5mb.txt")

# Basic settings
VAL_RATIO = 0.1
CONTEXT_LENGTH = 64   # small safe prototype value

def main():
    # 1. Read raw text
    text = INPUT_FILE.read_text(encoding="utf-8")

    print("=== Training chunk preparation ===")
    print("Input file:", INPUT_FILE)
    print("Total characters in corpus:", len(text))

    # 2. Build vocabulary
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    print("Vocab size:", len(vocab))

    # 3. Encode full text
    encoded = [stoi[ch] for ch in text]

    print("Encoded sequence length:", len(encoded))

    # 4. Train / validation split
    split_idx = int(len(encoded) * (1 - VAL_RATIO))
    train_ids = encoded[:split_idx]
    val_ids = encoded[split_idx:]

    print("Train length:", len(train_ids))
    print("Validation length:", len(val_ids))
    print("Context length:", CONTEXT_LENGTH)

    # 5. Safety check
    if len(train_ids) <= CONTEXT_LENGTH:
        print("ERROR: train sequence is too short for this context length.")
        return

    # 6. Make one example chunk from training data
    start_idx = 0
    x = train_ids[start_idx : start_idx + CONTEXT_LENGTH]
    y = train_ids[start_idx + 1 : start_idx + CONTEXT_LENGTH + 1]

    # 7. Decode for readability
    x_text = "".join(itos[i] for i in x)
    y_text = "".join(itos[i] for i in y)

    # 8. Print checks
    print("\nLength of x:", len(x))
    print("Length of y:", len(y))

    print("\nFirst training chunk x (raw text):")
    print(x_text)

    print("\nFirst training chunk y (raw text):")
    print(y_text)

    print("\nFirst 30 ids of x:")
    print(x[:30])

    print("\nFirst 30 ids of y:")
    print(y[:30])

    # 9. Show position-by-position next-character prediction alignment
    print("\nFirst 20 position pairs:")
    for i in range(20):
        current_char = itos[x[i]]
        next_char = itos[y[i]]
        print(f"position {i}: input='{current_char}' -> target='{next_char}'")

if __name__ == "__main__":
    main()