from poetry_lm.analysis import compare_statistics, text_statistics


def test_structural_statistics_preserve_poem_and_line_boundaries() -> None:
    statistics = text_statistics("春风，来。\n明月。\n\n秋水。")
    assert statistics["num_poems"] == 2
    assert statistics["num_lines"] == 3
    assert statistics["line_length_distribution"] == {"2": 2, "3": 1}


def test_identical_text_has_zero_distribution_distance() -> None:
    statistics = text_statistics("春风。\n明月。")
    comparison = compare_statistics(statistics, statistics)
    assert all(value == 0 for value in comparison.values())
