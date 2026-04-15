from pathlib import Path
import random

# Input corpus
INPUT_FILE = Path("songci_pilot_1_5mb.txt")

# Settings
VAL_RATIO = 0.1
CONTEXT_LENGTH = 64
BATCH_SIZE = 4
RANDOM_SEED = 42

def prepare_encoded_data(input_file: Path, val_ratio=0.1):
    text = input_file.read_text(encoding="utf-8")

    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    encoded = [stoi[ch] for ch in text]

    split_idx = int(len(encoded) * (1 - val_ratio))
    train_ids = encoded[:split_idx]
    val_ids = encoded[split_idx:]

    return text, vocab, stoi, itos, encoded, train_ids, val_ids

def get_batch(split_ids, batch_size, context_length):
    """
    Randomly sample batch_size starting positions.
    For each start position:
      x = split_ids[start : start + context_length]
      y = split_ids[start + 1 : start + context_length + 1]
    """
    max_start = len(split_ids) - context_length - 1
    if max_start <= 0:
        raise ValueError("Sequence too short for the chosen context length.")

    starts = [random.randint(0, max_start) for _ in range(batch_size)]

    x_batch = []
    y_batch = []

    for start in starts:
        x = split_ids[start : start + context_length]
        y = split_ids[start + 1 : start + context_length + 1]
        x_batch.append(x)
        y_batch.append(y)

    return starts, x_batch, y_batch

def decode_ids(id_list, itos):
    return "".join(itos[i] for i in id_list)

def main():
    random.seed(RANDOM_SEED)

    text, vocab, stoi, itos, encoded, train_ids, val_ids = prepare_encoded_data(
        INPUT_FILE, VAL_RATIO
    )

    print("=== Mini-batch sampler ===")
    print("Input file:", INPUT_FILE)
    print("Total characters:", len(text))
    print("Vocab size:", len(vocab))
    print("Train length:", len(train_ids))
    print("Validation length:", len(val_ids))
    print("Context length:", CONTEXT_LENGTH)
    print("Batch size:", BATCH_SIZE)
    print("Random seed:", RANDOM_SEED)

    # Build one train batch
    train_starts, x_batch, y_batch = get_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH)

    print("\nTrain batch sampled successfully.")
    print("Start positions:", train_starts)

    print("\nBatch dimensions:")
    print("  Number of x chunks:", len(x_batch))
    print("  Number of y chunks:", len(y_batch))
    print("  Length of one x chunk:", len(x_batch[0]))
    print("  Length of one y chunk:", len(y_batch[0]))

    # Print first example in detail
    x0 = x_batch[0]
    y0 = y_batch[0]

    print("\nFirst batch example x (raw text):")
    print(decode_ids(x0, itos))

    print("\nFirst batch example y (raw text):")
    print(decode_ids(y0, itos))

    print("\nFirst 30 ids of x0:")
    print(x0[:30])

    print("\nFirst 30 ids of y0:")
    print(y0[:30])

    print("\nFirst 20 position pairs from first example:")
    for i in range(20):
        current_char = itos[x0[i]]
        next_char = itos[y0[i]]
        print(f"position {i}: input='{current_char}' -> target='{next_char}'")

    # Print short preview of all examples
    print("\nShort preview of all batch examples:")
    for i in range(BATCH_SIZE):
        preview = decode_ids(x_batch[i][:30], itos).replace("\n", "\\n")
        print(f"  Example {i}: start={train_starts[i]}, preview={preview}")

if __name__ == "__main__":
    main()