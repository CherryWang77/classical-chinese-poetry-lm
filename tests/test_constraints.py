from __future__ import annotations

import torch

from poetry_lm.constraints import RhymeLexicon, SongCiAutomaton
from poetry_lm.research_types import SegmentSpec, TemplateSpec
from poetry_lm.templates import render_with_template_breaks
from poetry_lm.token_constraints import ConstrainedLogitsProcessor, TokenSurfaceTable


class MockTokenizer:
    surfaces = (
        "<eos>",
        "山",
        "河，\n",
        "月。\n\n",
        "山河，\n",
        "\ufffd",
        "\t",
        "坏",
        "天。\n\n",
        "江。\n\n",
        "，\n",
    )

    def __len__(self) -> int:
        return len(self.surfaces)

    @property
    def all_special_ids(self) -> list[int]:
        return [0]

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        skip = bool(kwargs.get("skip_special_tokens"))
        return "".join(
            surface
            for token_id in token_ids
            if not (skip and token_id == 0)
            for surface in [self.surfaces[token_id]]
        )

    def encode(self, text: str, **kwargs: object) -> list[int]:
        if text == "坏":
            return [7, 1]
        try:
            return [self.surfaces.index(text)]
        except ValueError:
            return []


def structure_template() -> TemplateSpec:
    return TemplateSpec(
        template_id="测试:v0",
        cipai="测试",
        variant=0,
        source_commit="fixture",
        segments=(
            SegmentSpec(char_count=2, punctuation="，"),
            SegmentSpec(char_count=1, punctuation="。", stanza_end=True),
        ),
    )


def rhyme_template() -> TemplateSpec:
    return TemplateSpec(
        template_id="押韵:v0",
        cipai="押韵",
        variant=0,
        source_commit="fixture",
        segments=(
            SegmentSpec(
                char_count=1,
                punctuation="，",
                rhyme_slot=True,
                rhyme_chain=0,
            ),
            SegmentSpec(
                char_count=1,
                punctuation="。",
                stanza_end=True,
                rhyme_slot=True,
                rhyme_chain=0,
            ),
        ),
    )


def test_automaton_accepts_only_complete_structure() -> None:
    automaton = SongCiAutomaton(structure_template())
    canonical = "山河，\n月。\n\n"
    assert automaton.is_accepting(automaton.require_complete(canonical))
    rejected = (
        "山，\n月。\n\n",
        "山河海，\n月。\n\n",
        "山河。\n月。\n\n",
        "山河，\n\n月。\n\n",
        "山河，\n月。\n",
    )
    for text in rejected:
        assert automaton.consume(text) is None or not automaton.is_accepting(
            automaton.consume(text).state  # type: ignore[union-attr]
        )


def test_corpus_text_can_be_rendered_into_canonical_fsa_layout() -> None:
    template = structure_template()
    rendered = render_with_template_breaks("山河，\n月。", template)
    assert rendered == "山河，\n月。\n\n"
    SongCiAutomaton(template).require_complete(rendered)


def test_multi_character_token_can_cross_punctuation() -> None:
    automaton = SongCiAutomaton(structure_template())
    first = automaton.transition(automaton.initial_state, "山河，\n")
    assert first is not None
    assert first.state.offset == 4
    second = automaton.transition(first.state, "月。\n\n")
    assert second is not None and automaton.is_accepting(second.state)


def test_token_surface_filter_and_eos_masking() -> None:
    tokenizer = MockTokenizer()
    table = TokenSurfaceTable.from_tokenizer(tokenizer)
    kept = {surface.token_id for surface in table.surfaces}
    assert 0 not in kept
    assert 5 not in kept
    assert 6 not in kept
    assert 7 not in kept

    processor = ConstrainedLogitsProcessor(
        tokenizer,
        SongCiAutomaton(structure_template()),
        prompt_length=0,
        eos_token_id=0,
        surface_table=table,
    )
    scores = torch.zeros((1, len(tokenizer)))
    initial = processor(torch.empty((1, 0), dtype=torch.long), scores)
    assert torch.isneginf(initial[0, 0])
    after_first = processor(torch.tensor([[4]]), scores)
    assert not torch.isneginf(after_first[0, 3])
    accepting = processor(torch.tensor([[4, 3]]), scores)
    assert accepting[0, 0] == 0
    assert torch.isneginf(accepting[0, 1:]).all()


def test_rhyme_reward_is_soft_and_unmapped_characters_do_not_crash() -> None:
    lexicon = RhymeLexicon(
        {
            "山": frozenset({"第一部"}),
            "天": frozenset({"第一部"}),
            "江": frozenset({"第二部"}),
        }
    )
    automaton = SongCiAutomaton(rhyme_template(), lexicon)
    prefix = automaton.consume("山，\n")
    assert prefix is not None
    matching = automaton.transition(prefix.state, "天。\n\n")
    mismatching = automaton.transition(prefix.state, "江。\n\n")
    unmapped = automaton.transition(prefix.state, "月。\n\n")
    assert matching is not None and matching.rhyme_reward_events == 1
    assert mismatching is not None and mismatching.rhyme_reward_events == 0
    assert unmapped is not None and unmapped.rhyme_reward_events == 0

    tokenizer = MockTokenizer()
    processor = ConstrainedLogitsProcessor(
        tokenizer,
        automaton,
        prompt_length=0,
        eos_token_id=0,
        rhyme_lambda=2.0,
    )
    output = processor(torch.tensor([[1]]), torch.zeros((1, len(tokenizer))))
    # The single-token prefix has not consumed its required comma, so continuation
    # tokens beginning with punctuation are the only legal set at this state.
    assert torch.isneginf(output[0, 8])


def test_rhyme_transition_reward_changes_legal_token_score() -> None:
    tokenizer = MockTokenizer()
    automaton = SongCiAutomaton(
        rhyme_template(),
        RhymeLexicon(
            {
                "山": frozenset({"第一部"}),
                "天": frozenset({"第一部"}),
                "江": frozenset({"第二部"}),
            }
        ),
    )
    # Add a surface for the first complete segment so the second segment begins
    # exactly at the next decoding step.
    tokenizer.surfaces = (*tokenizer.surfaces, "山，\n")
    processor = ConstrainedLogitsProcessor(
        tokenizer,
        automaton,
        prompt_length=0,
        eos_token_id=0,
        rhyme_lambda=2.0,
    )
    prefix_id = len(tokenizer) - 1
    output = processor(
        torch.tensor([[prefix_id]]), torch.zeros((1, len(tokenizer)))
    )
    assert output[0, 8] == 2.0
    assert output[0, 9] == 0.0
