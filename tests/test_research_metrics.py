from poetry_lm.research_metrics import (
    choose_rhyme_lambda,
    paired_bootstrap_difference,
    structural_metrics,
    wilson_interval,
)
from poetry_lm.research_types import SegmentSpec, TemplateSpec


def template() -> TemplateSpec:
    return TemplateSpec(
        template_id="测试:v0",
        cipai="测试",
        variant=0,
        source_commit="fixture",
        segments=(
            SegmentSpec(char_count=1, punctuation="，", stanza_end=True),
            SegmentSpec(char_count=1, punctuation="。", stanza_end=True),
        ),
    )


def test_structural_exact_match_includes_stanza_boundary() -> None:
    correct = structural_metrics("山，\n\n水。\n\n", template())
    wrong = structural_metrics("山，\n水。\n\n", template())
    assert correct["structural_exact_match"] is True
    assert wrong["segment_length_accuracy"] == 1.0
    assert wrong["punctuation_accuracy"] == 1.0
    assert wrong["structural_exact_match"] is False


def test_lambda_rule_and_statistical_intervals_are_deterministic() -> None:
    rows = {
        0.0: {"distinct_2": 1.0, "repetition_rate": 0.01, "rhyme_consistency": 0.4},
        0.5: {"distinct_2": 0.98, "repetition_rate": 0.02, "rhyme_consistency": 0.7},
        1.0: {"distinct_2": 0.96, "repetition_rate": 0.03, "rhyme_consistency": 0.7},
        2.0: {"distinct_2": 0.94, "repetition_rate": 0.02, "rhyme_consistency": 0.9},
    }
    assert choose_rhyme_lambda(rows) == 0.5
    first = paired_bootstrap_difference([1.0, 0.0], [0.0, 0.0], resamples=100)
    second = paired_bootstrap_difference([1.0, 0.0], [0.0, 0.0], resamples=100)
    assert first == second
    low, high = wilson_interval(10, 10)
    assert 0.7 < low < high <= 1.0
