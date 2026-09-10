from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .constraints import (
    CharacterSlot,
    ConstraintDeadEndError,
    ConstraintState,
    LiteralSlot,
    SongCiAutomaton,
)


class TokenizerProtocol(Protocol):
    def __len__(self) -> int: ...

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str: ...

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...

    @property
    def all_special_ids(self) -> list[int]: ...


@dataclass(frozen=True)
class TokenSurface:
    token_id: int
    text: str


class TokenSurfaceTable:
    def __init__(self, surfaces: tuple[TokenSurface, ...]):
        self.surfaces = surfaces
        first: dict[str, list[TokenSurface]] = defaultdict(list)
        han: list[TokenSurface] = []
        for surface in surfaces:
            first[surface.text[0]].append(surface)
            if _starts_with_han(surface.text):
                han.append(surface)
        self.by_first_character = {key: tuple(value) for key, value in first.items()}
        self.han_surfaces = tuple(han)

    @classmethod
    def from_tokenizer(cls, tokenizer: TokenizerProtocol) -> TokenSurfaceTable:
        special_ids = set(int(value) for value in tokenizer.all_special_ids)
        surfaces: list[TokenSurface] = []
        for token_id in range(len(tokenizer)):
            if token_id in special_ids:
                continue
            text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if not text or "\ufffd" in text or "\r" in text or "\t" in text:
                continue
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if encoded != [token_id]:
                continue
            surfaces.append(TokenSurface(token_id, text))
        return cls(tuple(surfaces))

    def candidates(self, automaton: SongCiAutomaton, state: ConstraintState) -> tuple[TokenSurface, ...]:
        expected = automaton.next_expected(state)
        if isinstance(expected, CharacterSlot):
            return self.han_surfaces
        if isinstance(expected, LiteralSlot):
            return self.by_first_character.get(expected.value, ())
        return ()


def _starts_with_han(text: str) -> bool:
    from .constraints import is_han_character

    return bool(text) and is_han_character(text[0])


class ConstrainedLogitsProcessor:
    """Transformers-compatible online structural and rhyme constraint processor."""

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        automaton: SongCiAutomaton,
        prompt_length: int,
        eos_token_id: int,
        rhyme_lambda: float = 0.0,
        surface_table: TokenSurfaceTable | None = None,
    ):
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        if rhyme_lambda < 0:
            raise ValueError("rhyme_lambda must be non-negative")
        self.tokenizer = tokenizer
        self.automaton = automaton
        self.prompt_length = prompt_length
        self.eos_token_id = int(eos_token_id)
        self.rhyme_lambda = float(rhyme_lambda)
        self.surface_table = surface_table or TokenSurfaceTable.from_tokenizer(tokenizer)
        self._legal_cache: dict[
            ConstraintState, tuple[tuple[int, float], ...]
        ] = {}

    def state_from_ids(self, token_ids: Sequence[int]) -> ConstraintState:
        generated = self.tokenizer.decode(
            list(token_ids)[self.prompt_length :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not generated:
            return self.automaton.initial_state
        result = self.automaton.consume(generated)
        if result is None:
            raise ConstraintDeadEndError(
                f"generated prefix violates {self.automaton.template.template_id}: {generated!r}"
            )
        return result.state

    def _legal(self, state: ConstraintState) -> tuple[tuple[int, float], ...]:
        if state in self._legal_cache:
            return self._legal_cache[state]
        candidates = self.surface_table.candidates(self.automaton, state)
        legal: list[tuple[int, float]] = []
        for surface in candidates:
            transition = self.automaton.transition(state, surface.text)
            if transition is not None:
                reward = self.rhyme_lambda * transition.rhyme_reward_events
                legal.append((surface.token_id, reward))
        if not legal:
            expected = self.automaton.next_expected(state)
            raise ConstraintDeadEndError(
                f"no legal tokenizer transition at offset {state.offset}; expected {expected!r}"
            )
        result = tuple(legal)
        self._legal_cache[state] = result
        return result

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or scores.ndim != 2:
            raise ValueError("input_ids and scores must both be rank-2 tensors")
        masked = torch.full_like(scores, float("-inf"))
        for row in range(input_ids.size(0)):
            state = self.state_from_ids(input_ids[row].tolist())
            if self.automaton.is_accepting(state):
                masked[row, self.eos_token_id] = scores[row, self.eos_token_id]
                continue
            legal = self._legal(state)
            token_ids = torch.tensor(
                [token_id for token_id, _ in legal], dtype=torch.long, device=scores.device
            )
            rewards = torch.tensor(
                [reward for _, reward in legal], dtype=scores.dtype, device=scores.device
            )
            masked[row, token_ids] = scores[row, token_ids] + rewards
        return masked
