from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# Files
INPUT_FILE = Path("songci_pilot_1_5mb.txt")
CHECKPOINT_PATH = Path("checkpoints_decoder_block/decoder_block_final.pt")

# Generation settings
CONTEXT_LENGTH = 64
EMBED_DIM = 128
PROMPT = "气和玉烛，睿化著鸿明。\n缇管一阳生。\n"
MAX_NEW_CHARS = 200
TEMPERATURE = 0.8
TOP_K = 20
RANDOM_SEED = 42

def prepare_vocab_from_corpus(input_file: Path):
    text = input_file.read_text(encoding="utf-8")
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    return text, vocab, stoi, itos

def encode_text(text: str, stoi: dict):
    ids = []
    for ch in text:
        if ch not in stoi:
            raise ValueError(f"Character not found in vocabulary: {repr(ch)}")
        ids.append(stoi[ch])
    return ids

def decode_ids(ids, itos: dict):
    return "".join(itos[i] for i in ids)

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

def sample_next_id(next_logits, temperature=1.0, top_k=None):
    # temperature scaling
    next_logits = next_logits / temperature

    # top-k filtering
    if top_k is not None and top_k > 0:
        values, indices = torch.topk(next_logits, k=top_k)
        filtered_logits = torch.full_like(next_logits, float("-inf"))
        filtered_logits[indices] = next_logits[indices]
        next_logits = filtered_logits

    probs = F.softmax(next_logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    return next_id

def generate_sampled(model, start_ids, max_new_chars, context_length, temperature=1.0, top_k=None):
    model.eval()
    generated = list(start_ids)

    with torch.no_grad():
        for _ in range(max_new_chars):
            x_cond = generated[-context_length:]
            x_tensor = torch.tensor([x_cond], dtype=torch.long)

            logits = model(x_tensor)          # (1, T, vocab_size)
            next_logits = logits[0, -1]       # (vocab_size,)

            next_id = sample_next_id(
                next_logits,
                temperature=temperature,
                top_k=top_k
            )
            generated.append(next_id)

    return generated

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("=== Generate from decoder block checkpoint (sampled) ===")
    print("Input corpus:", INPUT_FILE)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Prompt:", repr(PROMPT))
    print("Max new chars:", MAX_NEW_CHARS)
    print("Temperature:", TEMPERATURE)
    print("Top-k:", TOP_K)
    print("Random seed:", RANDOM_SEED)

    # 1. rebuild vocab
    corpus_text, vocab, stoi, itos = prepare_vocab_from_corpus(INPUT_FILE)
    vocab_size = len(vocab)
    print("Vocab size:", vocab_size)

    # 2. rebuild model
    model = MinimalDecoderLM(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        context_length=CONTEXT_LENGTH
    )

    # 3. load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Checkpoint loaded successfully.")
    print("Checkpoint step:", checkpoint["step"])

    # 4. encode prompt
    start_ids = encode_text(PROMPT, stoi)
    print("Prompt length (chars):", len(PROMPT))
    print("Prompt IDs:", start_ids)

    # 5. generate
    generated_ids = generate_sampled(
        model=model,
        start_ids=start_ids,
        max_new_chars=MAX_NEW_CHARS,
        context_length=CONTEXT_LENGTH,
        temperature=TEMPERATURE,
        top_k=TOP_K
    )

    generated_text = decode_ids(generated_ids, itos)

    print("\n=== Generated text ===")
    print(generated_text)

if __name__ == "__main__":
    main()