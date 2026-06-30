#!/usr/bin/env python3
from __future__ import annotations

"""Style-aware vowel confusion analysis across ASR experiments."""

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_tone_vowel_confusions import EPSILON, _analyze_rows, _read_predictions


VOWEL_LABELS = ["a", "e", "\u1eb9", "i", "o", "\u1ecd", "u", "an", "\u1eb9n", "\u1ecdn", "un", "in"]
MATRIX_LABELS = [EPSILON, *VOWEL_LABELS]
NATURALISTIC_EXPERIMENTS = ["E1", "E2", "E3", "E6", "E7", "E8"]
CONSTRAINED_EXPERIMENTS = ["E4", "E5", "E9", "E10", "E11", "E12"]
EXPERIMENT_DIRS = {
    "E1": "E1_nat2nat_main_noXfer",
    "E2": "E2_cons2nat_main_noXfer",
    "E3": "E3_mix2nat_main_noXfer",
    "E4": "E4_nat2cons_rev_noXfer",
    "E5": "E5_mix2cons_aux_noXfer",
    "E6": "E6_nat2nat_main_xfer",
    "E7": "E7_cons2nat_main_xfer",
    "E8": "E8_mix2nat_main_xfer",
    "E9": "E9_nat2cons_rev_xfer",
    "E10": "E10_mix2cons_aux_xfer",
    "E11": "E11_cons2cons_aux_noXfer",
    "E12": "E12_cons2cons_aux_xfer",
}


def _experiment_style(experiment: str) -> str:
    if experiment in NATURALISTIC_EXPERIMENTS:
        return "naturalistic"
    if experiment in CONSTRAINED_EXPERIMENTS:
        return "constrained"
    raise KeyError(f"Unknown experiment: {experiment}")


def _find_predictions(results_root: Path, experiment: str) -> Path:
    expected = results_root / EXPERIMENT_DIRS[experiment] / "predictions.csv"
    if expected.is_file():
        return expected

    matches = sorted(
        path
        for path in results_root.rglob("predictions.csv")
        if path.parent.name == EXPERIMENT_DIRS[experiment]
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No predictions.csv found for {experiment} ({EXPERIMENT_DIRS[experiment]})")


def _canonicalize_counts(counts: dict[str, Counter[str]]) -> dict[str, Counter[str]]:
    """Keep requested vowel categories and preserve eps insertions/deletions."""

    canonical_counts: dict[str, Counter[str]] = defaultdict(Counter)
    labels = set(MATRIX_LABELS)
    for reference, row in counts.items():
        canonical_reference = reference if reference in labels else reference
        for prediction, count in row.items():
            canonical_prediction = prediction if prediction in labels else prediction
            canonical_counts[canonical_reference][canonical_prediction] += count
    for label in MATRIX_LABELS:
        canonical_counts[label]
    return canonical_counts


def _ordered_matrix_labels(counts: dict[str, Counter[str]]) -> list[str]:
    labels = set(MATRIX_LABELS)
    labels.update(counts.keys())
    for row in counts.values():
        labels.update(row.keys())
    return [label for label in MATRIX_LABELS if label in labels] + sorted(label for label in labels if label not in MATRIX_LABELS)


def _row_total(counts: dict[str, Counter[str]], label: str, labels: list[str]) -> int:
    return sum(counts[label].get(prediction, 0) for prediction in labels)


def _accuracy(counts: dict[str, Counter[str]], label: str, labels: list[str]) -> float | None:
    total = _row_total(counts, label, labels)
    if total == 0:
        return None
    return counts[label].get(label, 0) / total


def _write_count_matrix(path: Path, counts: dict[str, Counter[str]], labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference\\prediction", *labels])
        for reference in labels:
            writer.writerow([reference, *[counts[reference].get(prediction, 0) for prediction in labels]])


def _write_row_normalized_matrix(path: Path, counts: dict[str, Counter[str]], labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference\\prediction", *labels])
        for reference in labels:
            total = _row_total(counts, reference, labels)
            writer.writerow(
                [
                    reference,
                    *[
                        counts[reference].get(prediction, 0) / total if total else 0.0
                        for prediction in labels
                    ],
                ]
            )


def _accuracy_rows(counts: dict[str, Counter[str]], labels: list[str], experiment: str, style: str) -> list[dict[str, object]]:
    rows = []
    for vowel in VOWEL_LABELS:
        total = _row_total(counts, vowel, labels)
        correct = counts[vowel].get(vowel, 0)
        acc = correct / total if total else None
        rows.append(
            {
                "experiment": experiment,
                "test_style": style,
                "vowel": vowel,
                "reference_count": total,
                "correct_count": correct,
                "accuracy": "" if acc is None else acc,
                "error_rate": "" if acc is None else 1.0 - acc,
            }
        )
    return rows


def _write_per_vowel_accuracy(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["experiment", "test_style", "vowel", "reference_count", "correct_count", "accuracy", "error_rate"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _hex_color(value: float) -> str:
    value = max(0.0, min(1.0, value))
    start = (247, 250, 252)
    end = (37, 99, 235)
    rgb = tuple(round(start[idx] + (end[idx] - start[idx]) * value) for idx in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _write_heatmap_svg(
    path: Path,
    title: str,
    counts: dict[str, Counter[str]],
    labels: list[str],
    normalized: bool,
) -> None:
    cell = 42
    left = 92
    top = 100
    width = left + cell * len(labels) + 35
    height = top + cell * len(labels) + 70
    max_count = max((counts[row].get(col, 0) for row in labels for col in labels), default=1)

    cells = []
    for row_idx, reference in enumerate(labels):
        total = _row_total(counts, reference, labels)
        for col_idx, prediction in enumerate(labels):
            count = counts[reference].get(prediction, 0)
            value = count / total if normalized and total else count
            intensity = value if normalized else count / max(max_count, 1)
            x = left + col_idx * cell
            y = top + row_idx * cell
            text = f"{value:.2f}" if normalized else str(count)
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{_hex_color(intensity)}" stroke="#ffffff"/>'
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" text-anchor="middle" class="cell-text">{html.escape(text)}</text>'
            )

    x_labels = [
        f'<text x="{left + idx * cell + cell / 2}" y="{top - 10}" text-anchor="middle" class="tick">{html.escape(label)}</text>'
        for idx, label in enumerate(labels)
    ]
    y_labels = [
        f'<text x="{left - 10}" y="{top + idx * cell + cell / 2 + 4}" text-anchor="end" class="tick">{html.escape(label)}</text>'
        for idx, label in enumerate(labels)
    ]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <rect width="{width}" height="{height}" fill="#ffffff"/>
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: #1f2933; }}
    .title {{ font-size: 18px; font-weight: 700; }}
    .axis {{ font-size: 13px; font-weight: 700; fill: #52606d; }}
    .tick {{ font-size: 11px; font-weight: 700; }}
    .cell-text {{ font-size: 10px; font-weight: 700; fill: #111827; }}
  </style>
  <text x="24" y="32" class="title">{html.escape(title)}</text>
  <text x="{left + cell * len(labels) / 2}" y="66" text-anchor="middle" class="axis">Predicted vowel</text>
  <text x="24" y="{top + cell * len(labels) / 2}" class="axis" transform="rotate(-90 24 {top + cell * len(labels) / 2})">Reference vowel</text>
  {"".join(x_labels)}
  {"".join(y_labels)}
  {"".join(cells)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def _add_counts(target: dict[str, Counter[str]], source: dict[str, Counter[str]]) -> None:
    for reference, row in source.items():
        target[reference].update(row)


def _write_experiment_accuracy_table(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["experiment", "test_style", "vowel", "reference_count", "correct_count", "accuracy", "error_rate"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_style_comparison(
    output_dir: Path,
    naturalistic_counts: dict[str, Counter[str]],
    constrained_counts: dict[str, Counter[str]],
    labels: list[str],
) -> None:
    comparison_rows = []
    for vowel in VOWEL_LABELS:
        naturalistic_accuracy = _accuracy(naturalistic_counts, vowel, labels)
        constrained_accuracy = _accuracy(constrained_counts, vowel, labels)
        if naturalistic_accuracy is None or constrained_accuracy is None:
            difference = ""
            degradation = ""
            abs_difference = ""
        else:
            difference = naturalistic_accuracy - constrained_accuracy
            degradation = constrained_accuracy - naturalistic_accuracy
            abs_difference = abs(difference)
        comparison_rows.append(
            {
                "vowel": vowel,
                "naturalistic_test_accuracy": "" if naturalistic_accuracy is None else naturalistic_accuracy,
                "constrained_test_accuracy": "" if constrained_accuracy is None else constrained_accuracy,
                "difference_naturalistic_minus_constrained": difference,
                "naturalistic_degradation_constrained_minus_naturalistic": degradation,
                "absolute_style_difference": abs_difference,
                "naturalistic_reference_count": _row_total(naturalistic_counts, vowel, labels),
                "constrained_reference_count": _row_total(constrained_counts, vowel, labels),
            }
        )

    comparison_path = output_dir / "vowel_style_accuracy_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    ranked_rows = sorted(
        comparison_rows,
        key=lambda row: row["absolute_style_difference"] if row["absolute_style_difference"] != "" else -999,
        reverse=True,
    )
    ranked_path = output_dir / "vowels_ranked_by_style_sensitivity.csv"
    with ranked_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ranked_rows)

    degradation_rows = sorted(
        comparison_rows,
        key=lambda row: row["naturalistic_degradation_constrained_minus_naturalistic"]
        if row["naturalistic_degradation_constrained_minus_naturalistic"] != ""
        else -999,
        reverse=True,
    )
    degradation_path = output_dir / "vowels_ranked_by_naturalistic_degradation.csv"
    with degradation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(degradation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(degradation_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run style-aware vowel analysis across E1-E12")
    parser.add_argument("--results-root", default="results", help="Root directory containing experiment outputs")
    parser.add_argument(
        "--output-dir",
        default="results/style_aware_vowel_analysis",
        help="Directory for style-aware vowel analysis outputs",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    naturalistic_counts: dict[str, Counter[str]] = defaultdict(Counter)
    constrained_counts: dict[str, Counter[str]] = defaultdict(Counter)
    all_accuracy_rows: list[dict[str, object]] = []
    manifest = []

    for experiment in [*NATURALISTIC_EXPERIMENTS, *CONSTRAINED_EXPERIMENTS]:
        style = _experiment_style(experiment)
        predictions_csv = _find_predictions(results_root, experiment)
        rows = _read_predictions(predictions_csv)
        analysis = _analyze_rows(rows)
        counts = _canonicalize_counts(analysis["vowel_counts"])
        labels = _ordered_matrix_labels(counts)
        experiment_dir = output_dir / experiment
        experiment_dir.mkdir(parents=True, exist_ok=True)

        _write_count_matrix(experiment_dir / "vowel_confusion_matrix.csv", counts, labels)
        _write_row_normalized_matrix(experiment_dir / "vowel_confusion_matrix_row_normalized.csv", counts, labels)
        accuracy_rows = _accuracy_rows(counts, labels, experiment, style)
        _write_per_vowel_accuracy(experiment_dir / "per_vowel_accuracy.csv", accuracy_rows)
        _write_heatmap_svg(experiment_dir / "vowel_confusion_heatmap.svg", f"{experiment} Vowel Confusion Matrix", counts, labels, normalized=False)
        _write_heatmap_svg(
            experiment_dir / "vowel_confusion_heatmap_row_normalized.svg",
            f"{experiment} Row-Normalized Vowel Confusion",
            counts,
            labels,
            normalized=True,
        )

        if style == "naturalistic":
            _add_counts(naturalistic_counts, counts)
        else:
            _add_counts(constrained_counts, counts)

        all_accuracy_rows.extend(accuracy_rows)
        manifest.append(
            {
                "experiment": experiment,
                "test_style": style,
                "predictions_csv": str(predictions_csv),
                "output_dir": str(experiment_dir),
            }
        )

    aggregate_labels = _ordered_matrix_labels({**naturalistic_counts, **constrained_counts})
    _write_count_matrix(output_dir / "naturalistic_test_vowel_confusion_matrix.csv", naturalistic_counts, aggregate_labels)
    _write_row_normalized_matrix(
        output_dir / "naturalistic_test_vowel_confusion_matrix_row_normalized.csv",
        naturalistic_counts,
        aggregate_labels,
    )
    _write_count_matrix(output_dir / "constrained_test_vowel_confusion_matrix.csv", constrained_counts, aggregate_labels)
    _write_row_normalized_matrix(
        output_dir / "constrained_test_vowel_confusion_matrix_row_normalized.csv",
        constrained_counts,
        aggregate_labels,
    )
    _write_heatmap_svg(
        output_dir / "naturalistic_test_vowel_confusion_heatmap.svg",
        "Naturalistic-Test Aggregate Vowel Confusion Matrix",
        naturalistic_counts,
        aggregate_labels,
        normalized=False,
    )
    _write_heatmap_svg(
        output_dir / "naturalistic_test_vowel_confusion_heatmap_row_normalized.svg",
        "Naturalistic-Test Row-Normalized Vowel Confusion",
        naturalistic_counts,
        aggregate_labels,
        normalized=True,
    )
    _write_heatmap_svg(
        output_dir / "constrained_test_vowel_confusion_heatmap.svg",
        "Constrained-Test Aggregate Vowel Confusion Matrix",
        constrained_counts,
        aggregate_labels,
        normalized=False,
    )
    _write_heatmap_svg(
        output_dir / "constrained_test_vowel_confusion_heatmap_row_normalized.svg",
        "Constrained-Test Row-Normalized Vowel Confusion",
        constrained_counts,
        aggregate_labels,
        normalized=True,
    )
    _write_experiment_accuracy_table(output_dir / "per_experiment_vowel_accuracy.csv", all_accuracy_rows)
    _write_style_comparison(output_dir, naturalistic_counts, constrained_counts, aggregate_labels)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved style-aware vowel analysis to: {output_dir}")


if __name__ == "__main__":
    main()
