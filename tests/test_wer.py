from offline_stt.wer import word_error_rate


def test_identical_ignoring_case_and_punctuation():
    assert word_error_rate("The birch canoe slid.", "the Birch canoe slid") == 0.0


def test_one_substitution_in_four_words():
    assert word_error_rate("glue the sheet down", "glue a sheet down") == 0.25


def test_insertions_and_deletions():
    assert word_error_rate("a b c", "a c") == 1 / 3
    assert word_error_rate("a b", "a x b y") == 1.0
