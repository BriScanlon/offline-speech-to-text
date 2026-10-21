"""Word error rate, for scoring a mic test against a known passage."""

import re

_WORD = re.compile(r"[a-z0-9']+")


def normalise(text: str) -> list[str]:
    """Lower-case words with punctuation removed, so only wording is scored."""
    return _WORD.findall(text.lower().replace("’", "'"))


def word_error_rate(reference: str, hypothesis: str) -> float:
    """(substitutions + deletions + insertions) / words in the reference."""
    ref = normalise(reference)
    hyp = normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)
