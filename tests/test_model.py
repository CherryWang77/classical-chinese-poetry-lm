import torch

from poetry_lm.config import ModelConfig
from poetry_lm.model import DecoderOnlyLM


def tiny_config(backend: str) -> ModelConfig:
    return ModelConfig(
        context_length=8,
        embed_dim=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        attention_backend=backend,
    )


def test_manual_and_sdpa_match_in_evaluation_mode() -> None:
    torch.manual_seed(7)
    manual = DecoderOnlyLM(vocab_size=23, config=tiny_config("manual"))
    sdpa = DecoderOnlyLM(vocab_size=23, config=tiny_config("sdpa"))
    sdpa.load_state_dict(manual.state_dict())
    manual.eval()
    sdpa.eval()
    tokens = torch.randint(0, 23, (2, 8))

    torch.testing.assert_close(manual(tokens), sdpa(tokens), rtol=1e-5, atol=1e-6)


def test_future_tokens_do_not_change_earlier_logits() -> None:
    torch.manual_seed(11)
    model = DecoderOnlyLM(vocab_size=23, config=tiny_config("manual"))
    model.eval()
    first = torch.randint(0, 23, (1, 8))
    second = first.clone()
    second[:, 5:] = torch.randint(0, 23, (1, 3))

    first_logits = model(first)
    second_logits = model(second)
    torch.testing.assert_close(first_logits[:, :5], second_logits[:, :5])


def test_sequence_longer_than_context_is_rejected() -> None:
    model = DecoderOnlyLM(vocab_size=23, config=tiny_config("manual"))
    tokens = torch.randint(0, 23, (1, 9))
    try:
        model(tokens)
    except ValueError as error:
        assert "exceeds context length" in str(error)
    else:
        raise AssertionError("expected a context-length error")
