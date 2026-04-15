from pathlib import Path
import random
import torch

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

def get_batch_as_tensors(split_ids, batch_size, context_length):
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

    x_tensor = torch.tensor(x_batch, dtype=torch.long)
    y_tensor = torch.tensor(y_batch, dtype=torch.long)

    return starts, x_tensor, y_tensor

def decode_ids(id_list, itos):
    return "".join(itos[int(i)] for i in id_list)

def main():
    random.seed(RANDOM_SEED)

    text, vocab, stoi, itos, encoded, train_ids, val_ids = prepare_encoded_data(
        INPUT_FILE, VAL_RATIO
    )

    print("=== PyTorch mini-batch preparation ===")
    print("Input file:", INPUT_FILE)
    print("Total characters:", len(text))
    print("Vocab size:", len(vocab))
    print("Train length:", len(train_ids))
    print("Validation length:", len(val_ids))
    print("Context length:", CONTEXT_LENGTH)
    print("Batch size:", BATCH_SIZE)

    starts, x_tensor, y_tensor = get_batch_as_tensors(
        train_ids, BATCH_SIZE, CONTEXT_LENGTH
    )

    print("\nSampled start positions:", starts)

    print("\nTensor info:")
    print("x_tensor.shape:", tuple(x_tensor.shape))
    print("y_tensor.shape:", tuple(y_tensor.shape))
    print("x_tensor.dtype:", x_tensor.dtype)
    print("y_tensor.dtype:", y_tensor.dtype)

    # Show first example
    x0 = x_tensor[0]
    y0 = y_tensor[0]

    print("\nFirst example x tensor (first 30 ids):")
    print(x0[:30])

    print("\nFirst example y tensor (first 30 ids):")
    print(y0[:30])

    print("\nFirst example x decoded:")
    print(decode_ids(x0, itos))

    print("\nFirst example y decoded:")
    print(decode_ids(y0, itos))

    print("\nFirst 20 position pairs from first tensor example:")
    for i in range(20):
        current_char = itos[int(x0[i])]
        next_char = itos[int(y0[i])]
        print(f"position {i}: input='{current_char}' -> target='{next_char}'")

if __name__ == "__main__":
    main()