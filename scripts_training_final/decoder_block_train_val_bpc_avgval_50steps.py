from pathlib import Path
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Input corpus
INPUT_FILE = Path("songci_pilot_1_5mb.txt")

# Settings
VAL_RATIO = 0.1
CONTEXT_LENGTH = 64
BATCH_SIZE = 4
RANDOM_SEED = 42
EMBED_DIM = 128
LEARNING_RATE = 1e-3
NUM_STEPS = 50
EVAL_EVERY = 5
NUM_VAL_BATCHES = 5

def loss_to_bpc(loss_value: float) -> float:
    return loss_value / math.log(2)

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

class SingleHeadCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1) / (C ** 0.5)

        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ V
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

class MinimalDecoderBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = SingleHeadCausalSelfAttention(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MinimalDecoderLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_length):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)
        self.block = MinimalDecoderBlock(embed_dim)
        self.final_ln = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape

        tok_emb = self.token_embedding(x)
        pos_ids = torch.arange(T, device=x.device)
        pos_emb = self.position_embedding(pos_ids)

        h = tok_emb + pos_emb
        h = self.block(h)
        h = self.final_ln(h)

        logits = self.output_projection(h)
        return logits

def compute_loss(model, x_tensor, y_tensor):
    logits = model(x_tensor)
    B, T, C = logits.shape
    logits_flat = logits.view(B * T, C)
    targets_flat = y_tensor.view(B * T)
    loss = F.cross_entropy(logits_flat, targets_flat)
    return loss

def evaluate_average_val_loss(model, val_ids, batch_size, context_length, num_val_batches):
    losses = []

    model.eval()
    with torch.no_grad():
        for _ in range(num_val_batches):
            _, x_val, y_val = get_batch_as_tensors(val_ids, batch_size, context_length)
            val_loss = compute_loss(model, x_val, y_val)
            losses.append(val_loss.item())

    avg_val_loss = sum(losses) / len(losses)
    return avg_val_loss, losses

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    text, vocab, stoi, itos, encoded, train_ids, val_ids = prepare_encoded_data(
        INPUT_FILE, VAL_RATIO
    )

    vocab_size = len(vocab)

    print("=== Decoder block train + avg validation + BPC loop (50 steps) ===")
    print("Input file:", INPUT_FILE)
    print("Vocab size:", vocab_size)
    print("Train length:", len(train_ids))
    print("Validation length:", len(val_ids))
    print("Context length:", CONTEXT_LENGTH)
    print("Batch size:", BATCH_SIZE)
    print("Embedding dim:", EMBED_DIM)
    print("Learning rate:", LEARNING_RATE)
    print("Num steps:", NUM_STEPS)
    print("Eval every:", EVAL_EVERY)
    print("Num val batches per eval:", NUM_VAL_BATCHES)

    model = MinimalDecoderLM(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        context_length=CONTEXT_LENGTH
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_loss_history = []
    train_bpc_history = []
    val_loss_history = []
    val_bpc_history = []

    for step in range(1, NUM_STEPS + 1):
        _, x_train, y_train = get_batch_as_tensors(
            train_ids, BATCH_SIZE, CONTEXT_LENGTH
        )

        model.train()
        optimizer.zero_grad()

        train_loss = compute_loss(model, x_train, y_train)
        train_loss.backward()
        optimizer.step()

        train_loss_value = train_loss.item()
        train_bpc_value = loss_to_bpc(train_loss_value)

        train_loss_history.append(train_loss_value)
        train_bpc_history.append(train_bpc_value)

        if step % EVAL_EVERY == 0:
            avg_val_loss, raw_val_losses = evaluate_average_val_loss(
                model, val_ids, BATCH_SIZE, CONTEXT_LENGTH, NUM_VAL_BATCHES
            )
            avg_val_bpc = loss_to_bpc(avg_val_loss)

            val_loss_history.append(avg_val_loss)
            val_bpc_history.append(avg_val_bpc)

            print(
                f"Step {step:02d} | "
                f"train loss = {train_loss_value:.6f} | train BPC = {train_bpc_value:.6f} | "
                f"avg val loss = {avg_val_loss:.6f} | avg val BPC = {avg_val_bpc:.6f}"
            )
        else:
            print(
                f"Step {step:02d} | "
                f"train loss = {train_loss_value:.6f} | train BPC = {train_bpc_value:.6f}"
            )

    print("\nDecoder block 50-step avg-validation + BPC loop completed successfully.")

    print("\nTrain loss summary:")
    print("First train loss:", train_loss_history[0])
    print("Last train loss:", train_loss_history[-1])
    print("Min train loss:", min(train_loss_history))
    print("Max train loss:", max(train_loss_history))
    print("Average train loss:", sum(train_loss_history) / len(train_loss_history))

    print("\nTrain BPC summary:")
    print("First train BPC:", train_bpc_history[0])
    print("Last train BPC:", train_bpc_history[-1])
    print("Min train BPC:", min(train_bpc_history))
    print("Max train BPC:", max(train_bpc_history))
    print("Average train BPC:", sum(train_bpc_history) / len(train_bpc_history))

    if val_loss_history:
        print("\nAverage validation loss summary:")
        print("First avg val loss:", val_loss_history[0])
        print("Last avg val loss:", val_loss_history[-1])
        print("Min avg val loss:", min(val_loss_history))
        print("Max avg val loss:", max(val_loss_history))
        print("Average of avg val losses:", sum(val_loss_history) / len(val_loss_history))

    if val_bpc_history:
        print("\nAverage validation BPC summary:")
        print("First avg val BPC:", val_bpc_history[0])
        print("Last avg val BPC:", val_bpc_history[-1])
        print("Min avg val BPC:", min(val_bpc_history))
        print("Max avg val BPC:", max(val_bpc_history))
        print("Average of avg val BPCs:", sum(val_bpc_history) / len(val_bpc_history))

if __name__ == "__main__":
    main()