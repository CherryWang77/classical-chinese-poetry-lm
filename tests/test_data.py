import random

import torch

from poetry_lm.data import batch_from_starts, fixed_validation_starts, sample_starts


def test_fixed_validation_starts_are_reproducible() -> None:
    ids = list(range(200))
    first = fixed_validation_starts(ids, 4, 16, 3, seed=1729)
    second = fixed_validation_starts(ids, 4, 16, 3, seed=1729)
    assert first == second


def test_batch_targets_are_shifted_by_one() -> None:
    ids = list(range(30))
    features, targets = batch_from_starts(ids, [2, 5], 4, torch.device("cpu"))
    assert features.tolist() == [[2, 3, 4, 5], [5, 6, 7, 8]]
    assert targets.tolist() == [[3, 4, 5, 6], [6, 7, 8, 9]]


def test_training_sampler_uses_its_own_rng() -> None:
    ids = list(range(200))
    first_rng = random.Random(42)
    second_rng = random.Random(42)
    assert sample_starts(ids, 8, 16, first_rng) == sample_starts(ids, 8, 16, second_rng)
