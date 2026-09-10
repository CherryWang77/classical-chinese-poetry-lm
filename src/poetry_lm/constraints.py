from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .research_types import TemplateSpec


class ConstraintDeadEndError(RuntimeError):
    """Raised instead of silently falling back to unconstrained generation."""


def is_han_character(character: str) -> bool:
    if len(character) != 1:
        return False
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x323AF
        or character == "〇"
    )


class RhymeLexicon:
    def __init__(self, mapping: dict[str, frozenset[str]]):
        self.mapping = mapping

    @classmethod
    def from_json(cls, path: Path) -> RhymeLexicon:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping: dict[str, frozenset[str]] = {}
        for character, row in payload.items():
            if isinstance(row, dict):
                rhyme = row.get("rhyme")
                values = rhyme if isinstance(rhyme, list) else [rhyme]
            elif isinstance(row, list):
                values = [item.get("rhyme") for item in row if isinstance(item, dict)]
            else:
                values = []
            groups = frozenset(str(value) for value in values if value)
            if groups:
                mapping[str(character)] = groups
        return cls(mapping)

    def groups(self, character: str) -> frozenset[str]:
        return self.mapping.get(character, frozenset())


@dataclass(frozen=True)
class CharacterSlot:
    segment_index: int
    position_in_segment: int
    rhyme_chain: int | None = None


@dataclass(frozen=True)
class LiteralSlot:
    value: str


Slot = CharacterSlot | LiteralSlot
RhymeAssignments = tuple[tuple[int, tuple[str, ...]], ...]


@dataclass(frozen=True)
class ConstraintState:
    offset: int = 0
    rhyme_assignments: RhymeAssignments = ()

    def assignment_dict(self) -> dict[int, frozenset[str]]:
        return {
            chain: frozenset(groups)
            for chain, groups in self.rhyme_assignments
        }


@dataclass(frozen=True)
class TransitionResult:
    state: ConstraintState
    rhyme_reward_events: int
    rhyme_slots_seen: int


class SongCiAutomaton:
    def __init__(self, template: TemplateSpec, rhyme_lexicon: RhymeLexicon | None = None):
        self.template = template
        self.rhyme_lexicon = rhyme_lexicon or RhymeLexicon({})
        slots: list[Slot] = []
        for segment_index, segment in enumerate(template.segments):
            for position in range(segment.char_count):
                rhyme_chain = (
                    segment.rhyme_chain
                    if segment.rhyme_slot and position == segment.char_count - 1
                    else None
                )
                slots.append(CharacterSlot(segment_index, position, rhyme_chain))
            slots.append(LiteralSlot(segment.punctuation))
            newline_count = 2 if segment.stanza_end else 1
            slots.extend(LiteralSlot("\n") for _ in range(newline_count))
        self.slots = tuple(slots)

    @property
    def initial_state(self) -> ConstraintState:
        return ConstraintState()

    def is_accepting(self, state: ConstraintState) -> bool:
        return state.offset == len(self.slots)

    def next_expected(self, state: ConstraintState) -> Slot | None:
        return None if self.is_accepting(state) else self.slots[state.offset]

    def transition(self, state: ConstraintState, surface: str) -> TransitionResult | None:
        if not surface:
            return None
        offset = state.offset
        assignments = state.assignment_dict()
        reward_events = 0
        rhyme_slots_seen = 0
        for character in surface:
            if offset >= len(self.slots):
                return None
            expected = self.slots[offset]
            if isinstance(expected, LiteralSlot):
                if character != expected.value:
                    return None
            else:
                if not is_han_character(character):
                    return None
                if expected.rhyme_chain is not None:
                    rhyme_slots_seen += 1
                    groups = self.rhyme_lexicon.groups(character)
                    chain = expected.rhyme_chain
                    if chain not in assignments:
                        if groups:
                            assignments[chain] = groups
                    elif groups and assignments[chain].intersection(groups):
                        reward_events += 1
            offset += 1
        serialized = tuple(
            (chain, tuple(sorted(groups)))
            for chain, groups in sorted(assignments.items())
        )
        return TransitionResult(
            ConstraintState(offset=offset, rhyme_assignments=serialized),
            rhyme_reward_events=reward_events,
            rhyme_slots_seen=rhyme_slots_seen,
        )

    def consume(self, text: str, state: ConstraintState | None = None) -> TransitionResult | None:
        return self.transition(state or self.initial_state, unicodedata.normalize("NFC", text))

    def require_complete(self, text: str) -> ConstraintState:
        result = self.consume(text)
        if result is None or not self.is_accepting(result.state):
            raise ValueError(f"text does not satisfy {self.template.template_id}")
        return result.state

    def legal_surfaces(
        self,
        state: ConstraintState,
        surfaces: Iterable[tuple[int, str]],
    ) -> list[tuple[int, TransitionResult]]:
        legal: list[tuple[int, TransitionResult]] = []
        for token_id, surface in surfaces:
            result = self.transition(state, surface)
            if result is not None:
                legal.append((token_id, result))
        return legal
