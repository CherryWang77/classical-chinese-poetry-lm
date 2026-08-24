from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class MultiHeadCausalSelfAttention(nn.Module):
    """Causal self-attention with interchangeable manual and PyTorch SDPA backends."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.backend = config.attention_backend

        self.query = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.key = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.value = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        query = self.query(x).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.key(x).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(x).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        if self.backend == "sdpa":
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = query @ key.transpose(-2, -1) / (self.head_dim**0.5)
            mask = self.causal_mask[:sequence_length, :sequence_length]
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = self.dropout(F.softmax(scores, dim=-1))
            output = weights @ value

        output = output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, channels
        )
        return self.out_proj(output)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.ReLU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadCausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class DecoderOnlyLM(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_layers)]
        )
        self.final_ln = nn.LayerNorm(config.embed_dim)
        self.output_projection = nn.Linear(config.embed_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = token_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length "
                f"{self.config.context_length}"
            )

        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(self.final_ln(hidden))


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
