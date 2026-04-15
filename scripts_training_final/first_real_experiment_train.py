from pathlib import Path
import math
import random
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Files and output paths
# =========================
INPUT_FILE = Path("songci_main_5mb.txt")
OUTPUT_CSV = Path("real_experiment_metrics.csv")
CHECKPOINT_DIR = Path("checkpoints_real_experiment")
SAMPLES_DIR = Path("samples_real_experiment")

# =========================
# First real experiment config
# =========================
VAL_RATIO = 0.1
CONTEXT_LENGTH = 256
BATCH_SIZE = 32
RANDOM_SEED = 42
EMBED_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.1
LEARNING_RATE = 3e-4
NUM_STEPS = 5000

LOG_EVERY = 100
EVAL_EVERY = 200
SAVE_EVERY = 1000
GENERATE_EVERY = 1000
NUM_VAL_BATCHES = 5

PROMPT = "气和玉烛，睿化著鸿明。\n缇管一阳生。\n"
SAMPLE_MAX_NEW_CHARS = 200
GEN_TEMPERATURE = 1.0
GEN_TOP_K = 50

# =========================
# Utilities
# =========================
def loss_to_bpc(loss_value: float) -> float:
    return loss_value / math.log(2)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

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

def encode_text(text: str, stoi: dict):
    ids = []
    for ch in text:
        if ch not in stoi:
            raise ValueError(f"Character not found in vocabulary: {repr(ch)}")
        ids.append(stoi[ch])
    return ids

def decode_ids(ids, itos: dict):
    return "".join(itos[i] for i in ids)

def get_batch_as_tensors(split_ids, batch_size, context_length, device):
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

    x_tensor = torch.tensor(x_batch, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y_batch, dtype=torch.long, device=device)

    return starts, x_tensor, y_tensor

# =========================
# Model components
# =========================
class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # reshape to heads
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)  # (B, H, T, T)

        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ V  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.out_proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadCausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class DecoderOnlyLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_length, num_layers, num_heads, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape

        tok_emb = self.token_embedding(x)
        pos_ids = torch.arange(T, device=x.device)
        pos_emb = self.position_embedding(pos_ids)

        h = tok_emb + pos_emb
        h = self.dropout(h)

        for block in self.blocks:
            h = block(h)

        h = self.final_ln(h)
        logits = self.output_projection(h)
        return logits

# =========================
# Training / evaluation helpers
# =========================
def compute_loss(model, x_tensor, y_tensor):
    logits = model(x_tensor)
    B, T, C = logits.shape
    logits_flat = logits.view(B * T, C)
    targets_flat = y_tensor.view(B * T)
    loss = F.cross_entropy(logits_flat, targets_flat)
    return loss

def evaluate_average_val_loss(model, val_ids, batch_size, context_length, num_val_batches, device):
    losses = []

    model.eval()
    with torch.no_grad():
        for _ in range(num_val_batches):
            _, x_val, y_val = get_batch_as_tensors(
                val_ids, batch_size, context_length, device
            )
            val_loss = compute_loss(model, x_val, y_val)
            losses.append(val_loss.item())

    avg_val_loss = sum(losses) / len(losses)
    return avg_val_loss, losses

def sample_next_id(next_logits, temperature=1.0, top_k=None):
    next_logits = next_logits / temperature

    if top_k is not None and top_k > 0:
        _, indices = torch.topk(next_logits, k=top_k)
        filtered_logits = torch.full_like(next_logits, float("-inf"))
        filtered_logits[indices] = next_logits[indices]
        next_logits = filtered_logits

    probs = F.softmax(next_logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    return next_id

def generate_sampled(model, start_ids, max_new_chars, context_length, temperature, top_k, device):
    model.eval()
    generated = list(start_ids)

    with torch.no_grad():
        for _ in range(max_new_chars):
            x_cond = generated[-context_length:]
            x_tensor = torch.tensor([x_cond], dtype=torch.long, device=device)

            logits = model(x_tensor)
            next_logits = logits[0, -1]

            next_id = sample_next_id(
                next_logits,
                temperature=temperature,
                top_k=top_k
            )
            generated.append(next_id)

    return generated

def write_csv_log(rows, output_csv: Path):
    fieldnames = [
        "step",
        "train_loss",
        "train_bpc",
        "avg_val_loss",
        "avg_val_bpc",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def save_checkpoint(path: Path, model, optimizer, step: int, vocab_size: int):
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "context_length": CONTEXT_LENGTH,
            "embed_dim": EMBED_DIM,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_steps": NUM_STEPS,
        },
    }
    torch.save(checkpoint, path)

def save_sample_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

# =========================
# Main
# =========================
def main():
    set_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    SAMPLES_DIR.mkdir(exist_ok=True)

    text, vocab, stoi, itos, encoded, train_ids, val_ids = prepare_encoded_data(
        INPUT_FILE, VAL_RATIO
    )
    vocab_size = len(vocab)

    print("=== First real experiment training script ===")
    print("Input file:", INPUT_FILE)
    print("Output CSV:", OUTPUT_CSV)
    print("Checkpoint dir:", CHECKPOINT_DIR)
    print("Samples dir:", SAMPLES_DIR)
    print("Device:", device)
    print("Vocab size:", vocab_size)
    print("Train length:", len(train_ids))
    print("Validation length:", len(val_ids))
    print("Context length:", CONTEXT_LENGTH)
    print("Batch size:", BATCH_SIZE)
    print("Embed dim:", EMBED_DIM)
    print("Num layers:", NUM_LAYERS)
    print("Num heads:", NUM_HEADS)
    print("Dropout:", DROPOUT)
    print("Learning rate:", LEARNING_RATE)
    print("Num steps:", NUM_STEPS)
    print("Log every:", LOG_EVERY)
    print("Eval every:", EVAL_EVERY)
    print("Save every:", SAVE_EVERY)
    print("Generate every:", GENERATE_EVERY)

    model = DecoderOnlyLM(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        context_length=CONTEXT_LENGTH,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start_ids = encode_text(PROMPT, stoi)

    train_loss_history = []
    train_bpc_history = []
    val_loss_history = []
    val_bpc_history = []
    csv_rows = []

    for step in range(1, NUM_STEPS + 1):
        _, x_train, y_train = get_batch_as_tensors(
            train_ids, BATCH_SIZE, CONTEXT_LENGTH, device
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

        row = {
            "step": step,
            "train_loss": train_loss_value,
            "train_bpc": train_bpc_value,
            "avg_val_loss": "",
            "avg_val_bpc": "",
        }

        if step % EVAL_EVERY == 0:
            avg_val_loss, _ = evaluate_average_val_loss(
                model, val_ids, BATCH_SIZE, CONTEXT_LENGTH, NUM_VAL_BATCHES, device
            )
            avg_val_bpc = loss_to_bpc(avg_val_loss)

            val_loss_history.append(avg_val_loss)
            val_bpc_history.append(avg_val_bpc)

            row["avg_val_loss"] = avg_val_loss
            row["avg_val_bpc"] = avg_val_bpc

        csv_rows.append(row)

        if step % LOG_EVERY == 0:
            if row["avg_val_loss"] != "":
                print(
                    f"Step {step:04d} | "
                    f"train loss = {train_loss_value:.6f} | train BPC = {train_bpc_value:.6f} | "
                    f"avg val loss = {row['avg_val_loss']:.6f} | avg val BPC = {row['avg_val_bpc']:.6f}"
                )
            else:
                print(
                    f"Step {step:04d} | "
                    f"train loss = {train_loss_value:.6f} | train BPC = {train_bpc_value:.6f}"
                )

        if step % SAVE_EVERY == 0:
            ckpt_path = CHECKPOINT_DIR / f"real_experiment_step_{step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step, vocab_size)
            print("Checkpoint saved to:", ckpt_path)

        if step % GENERATE_EVERY == 0:
            sample_ids = generate_sampled(
                model=model,
                start_ids=start_ids,
                max_new_chars=SAMPLE_MAX_NEW_CHARS,
                context_length=CONTEXT_LENGTH,
                temperature=GEN_TEMPERATURE,
                top_k=GEN_TOP_K,
                device=device
            )
            sample_text = decode_ids(sample_ids, itos)

            sample_path = SAMPLES_DIR / f"sample_step_{step}.txt"
            save_sample_text(sample_path, sample_text)
            print("Sample saved to:", sample_path)

    final_ckpt_path = CHECKPOINT_DIR / "real_experiment_final.pt"
    save_checkpoint(final_ckpt_path, model, optimizer, NUM_STEPS, vocab_size)

    write_csv_log(csv_rows, OUTPUT_CSV)

    print("\nFirst real experiment script completed.")
    print("Final checkpoint saved to:", final_ckpt_path)
    print("CSV log written to:", OUTPUT_CSV)

if __name__ == "__main__":
    main()