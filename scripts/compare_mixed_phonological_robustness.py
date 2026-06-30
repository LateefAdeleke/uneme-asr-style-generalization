#!/usr/bin/env python3
from __future__ import annotations

"""Compare constrained-only vs mixed-style training for phonological robustness."""

import argparse
import csv
from pathlib import Path


COMPARISONS = [
    {
        "comparison": "E11_vs_E5_no_transfer",
        "constrained_only": "E11",
        "mixed": "E5",
        "transfer_condition": "no_transfer",
    },
    {
        "comparison": "E12_vs_E10_transfer",
        "constrained_only": "E12",
        "mixed": "E10",
        "transfer_condition": "cross_lingual_transfer",
    },
]


def _read_metric_rows(path: Path, label_column: str) -> dict[str, dict[str, dict[str, str]]]:
    """Index per-experiment rows by experiment and tone/vowel label."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    indexed: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        indexed.setdefault(row["experiment"], {})[row[label_column]] = row
    return indexed


def _float_or_none(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def _int_or_zero(value: str) -> int:
    if value == "":
        return 0
    return int(value)


def _comparison_rows(
    indexed_rows: dict[str, dict[str, dict[str, str]]],
    label_column: str,
) -> list[dict[str, object]]:
    """Create comparison rows where positive improvement favors mixed training."""

    rows: list[dict[str, object]] = []
    labels = sorted({label for experiment_rows in indexed_rows.values() for label in experiment_rows})
    for comparison in COMPARISONS:
        constrained_experiment = comparison["constrained_only"]
        mixed_experiment = comparison["mixed"]
        for label in labels:
            constrained_row = indexed_rows.get(constrained_experiment, {}).get(label)
            mixed_row = indexed_rows.get(mixed_experiment, {}).get(label)
            if constrained_row is None or mixed_row is None:
                continue

            constrained_error = _float_or_none(constrained_row["error_rate"])
            mixed_error = _float_or_none(mixed_row["error_rate"])
            improvement = (
                None
                if constrained_error is None or mixed_error is None
                else constrained_error - mixed_error
            )
            rows.append(
                {
                    "comparison": comparison["comparison"],
                    "transfer_condition": comparison["transfer_condition"],
                    label_column: label,
                    "constrained_only_experiment": constrained_experiment,
                    "mixed_experiment": mixed_experiment,
                    "error_rate_constrained_only": "" if constrained_error is None else constrained_error,
                    "error_rate_mixed": "" if mixed_error is None else mixed_error,
                    "improvement": "" if improvement is None else improvement,
                    "constrained_only_reference_count": _int_or_zero(constrained_row["reference_count"]),
                    "mixed_reference_count": _int_or_zero(mixed_row["reference_count"]),
                    "constrained_only_correct_count": _int_or_zero(constrained_row["correct_count"]),
                    "mixed_correct_count": _int_or_zero(mixed_row["correct_count"]),
                }
            )
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: row["improvement"] if row["improvement"] != "" else -999,
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare mixed-style phonological robustness")
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
        default="results/mixed_phonological_robustness",
        help="Output directory for comparison CSVs",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    tone_rows = _comparison_rows(_read_metric_rows(Path(args.tone_accuracy), "tone"), "tone")
    vowel_rows = _comparison_rows(_read_metric_rows(Path(args.vowel_accuracy), "vowel"), "vowel")

    _write_rows(output_dir / "tone_mixed_training_improvement.csv", tone_rows)
    _write_rows(output_dir / "tone_mixed_training_improvement_ranked.csv", _rank_rows(tone_rows))
    _write_rows(output_dir / "vowel_mixed_training_improvement.csv", vowel_rows)
    _write_rows(output_dir / "vowel_mixed_training_improvement_ranked.csv", _rank_rows(vowel_rows))
    print(f"Saved mixed-training phonological robustness comparisons to: {output_dir}")


if __name__ == "__main__":
    main()
