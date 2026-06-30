#!/usr/bin/env python3
from __future__ import annotations

"""Measure phonological impact of matched vs mismatched speech-style training."""

import argparse
import csv
from pathlib import Path


MATCHED_EXPERIMENTS = ["E1", "E11"]
MISMATCHED_EXPERIMENTS = ["E2", "E4"]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _aggregate(rows: list[dict[str, str]], label_column: str, experiments: list[str]) -> dict[str, dict[str, int]]:
    aggregate: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["experiment"] not in experiments:
            continue
        label = row[label_column]
        label_counts = aggregate.setdefault(label, {"reference_count": 0, "correct_count": 0})
        label_counts["reference_count"] += int(row["reference_count"] or 0)
        label_counts["correct_count"] += int(row["correct_count"] or 0)
    return aggregate


def _error_rate(counts: dict[str, int]) -> float | None:
    total = counts["reference_count"]
    if total == 0:
        return None
    return 1.0 - (counts["correct_count"] / total)


def _comparison_rows(rows: list[dict[str, str]], label_column: str, category_type: str) -> list[dict[str, object]]:
    matched = _aggregate(rows, label_column, MATCHED_EXPERIMENTS)
    mismatched = _aggregate(rows, label_column, MISMATCHED_EXPERIMENTS)
    labels = sorted(set(matched) | set(mismatched))
    comparison_rows: list[dict[str, object]] = []
    for label in labels:
        matched_counts = matched.get(label, {"reference_count": 0, "correct_count": 0})
        mismatched_counts = mismatched.get(label, {"reference_count": 0, "correct_count": 0})
        match_error = _error_rate(matched_counts)
        mismatch_error = _error_rate(mismatched_counts)
        delta = None if match_error is None or mismatch_error is None else mismatch_error - match_error
        comparison_rows.append(
            {
                "category_type": category_type,
                "category": label,
                "matched_experiments": "+".join(MATCHED_EXPERIMENTS),
                "mismatched_experiments": "+".join(MISMATCHED_EXPERIMENTS),
                "match_error_rate": "" if match_error is None else match_error,
                "mismatch_error_rate": "" if mismatch_error is None else mismatch_error,
                "delta_error": "" if delta is None else delta,
                "matched_reference_count": matched_counts["reference_count"],
                "mismatched_reference_count": mismatched_counts["reference_count"],
                "matched_correct_count": matched_counts["correct_count"],
                "mismatched_correct_count": mismatched_counts["correct_count"],
            }
        )
    return comparison_rows


def _rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: row["delta_error"] if row["delta_error"] != "" else -999,
        reverse=True,
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure phonological impact of style mismatch")
    parser.add_argument(
        "--tone-accuracy",
        default="results/style_aware_tone_analysis/per_experiment_tone_accuracy.csv",
        help="Per-experiment tone accuracy table",
    )
    parser.add_argument(
        "--vowel-accuracy",
        default="results/style_aware_vowel_analysis/per_experiment_vowel_accuracy.csv",
        help="Per-experiment vowel accuracy table",
    )
    parser.add_argument(
        "--output-dir",
        default="results/style_mismatch_phonological_impact",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    tone_rows = _comparison_rows(_read_rows(Path(args.tone_accuracy)), "tone", "tone")
    vowel_rows = _comparison_rows(_read_rows(Path(args.vowel_accuracy)), "vowel", "vowel")
    combined_rows = [*tone_rows, *vowel_rows]

    _write_rows(output_dir / "tone_style_mismatch_impact.csv", tone_rows)
    _write_rows(output_dir / "tone_style_mismatch_impact_ranked.csv", _rank_rows(tone_rows))
    _write_rows(output_dir / "vowel_style_mismatch_impact.csv", vowel_rows)
    _write_rows(output_dir / "vowel_style_mismatch_impact_ranked.csv", _rank_rows(vowel_rows))
    _write_rows(output_dir / "phonological_style_mismatch_impact_ranked.csv", _rank_rows(combined_rows))
    print(f"Saved style-mismatch phonological impact tables to: {output_dir}")


if __name__ == "__main__":
    main()
