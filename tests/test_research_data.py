import pytest

from poetry_lm.research_data import UnionFind, verify_no_group_leakage
from poetry_lm.research_types import PoemRecord


def record(identifier: str, split: str, author: str, group_id: str) -> PoemRecord:
    return PoemRecord(
        id=identifier,
        author=author,
        rhythmic="测试",
        paragraphs=("山。",),
        text="山。",
        normalized_text="山。",
        template_id=None,
        template_exact=False,
        split=split,
        fold=0,
        source_file="fixture.json",
        source_index=0,
        sha256=identifier,
        group_id=group_id,
    )


def test_union_find_is_transitive() -> None:
    groups = UnionFind(4)
    groups.union(0, 1)
    groups.union(1, 2)
    assert groups.find(0) == groups.find(2)
    assert groups.find(3) != groups.find(0)


def test_group_leakage_is_rejected() -> None:
    rows = [
        record("a", "train", "甲", "group-a"),
        record("b", "test", "乙", "group-a"),
    ]
    with pytest.raises(AssertionError, match="union-find groups"):
        verify_no_group_leakage(rows)
