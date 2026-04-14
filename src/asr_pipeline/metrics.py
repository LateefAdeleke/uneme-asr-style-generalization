from __future__ import annotations

from typing import Iterable, List


def _levenshtein(a: List[str], b: List[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insertions = prev[j] + 1
            deletions = curr[j - 1] + 1
            substitutions = prev[j - 1] + (ca != cb)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]


def wer(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    refs = list(references)
    hyps = list(hypotheses)
    total_words = 0
    total_errs = 0
    for ref, hyp in zip(refs, hyps):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_words += len(ref_words)
        total_errs += _levenshtein(ref_words, hyp_words)
    return float(total_errs) / max(total_words, 1)


def cer(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    refs = list(references)
    hyps = list(hypotheses)
    total_chars = 0
    total_errs = 0
    for ref, hyp in zip(refs, hyps):
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        total_chars += len(ref_chars)
        total_errs += _levenshtein(ref_chars, hyp_chars)
    return float(total_errs) / max(total_chars, 1)
