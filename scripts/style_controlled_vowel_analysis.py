#!/usr/bin/env python3
from __future__ import annotations

"""Style-controlled vowel analysis for mixed-training ASR experiments.

This report isolates the experiments needed to test which vowel contrasts
account for the naturalistic-versus-constrained evaluation gap when training
composition is held constant:

- E3: mixed training -> naturalistic test, no transfer
- E5: mixed training -> constrained test, no transfer
- E8: transfer + mixed training -> naturalistic test
- E10: transfer + mixed training -> constrained test
"""

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_tone_vowel_confusions import EPSILON, _analyze_rows, _read_predictions


EXPERIMENTS = {
    "E3": {
        "directory": "E3_mix2nat_main_noXfer",
        "transfer": "No",
        "train_style": "mixed",
        "test_style": "naturalistic",
    },
    "E5": {
        "directory": "E5_mix2cons_aux_noXfer",
        "transfer": "No",
        "train_style": "mixed",
        "test_style": "constrained",
    },
    "E8": {
        "directory": "E8_mix2nat_main_xfer",
        "transfer": "Yes",
        "train_style": "mixed",
        "test_style": "naturalistic",
    },
    "E10": {
        "directory": "E10_mix2cons_aux_xfer",
        "transfer": "Yes",
        "train_style": "mixed",
        "test_style": "constrained",
    },
}

VOWEL_LABELS = ["a", "e", "ẹ", "i", "o", "ọ", "u", "an", "ẹn", "ọn", "un", "in"]
MATRIX_LABELS = [*VOWEL_LABELS, EPSILON]
ORAL_VOWELS = ["a", "e", "ẹ", "i", "o", "ọ", "u"]
NASAL_VOWELS = ["an", "ẹn", "ọn", "un", "in"]
VOWEL_CLASSES = {
    "front": ["e", "ẹ", "i", "ẹn", "in"],
    "central": ["a", "an"],
    "back": ["o", "ọ", "u", "ọn", "un"],
}
PAIRS = [
    ("E3_vs_E5", "E3", "E5", "no_transfer"),
    ("E8_vs_E10", "E8", "E10", "transfer"),
]


def _find_predictions(results_root: Path, experiment: str) -> Path:
    path = results_root / EXPERIMENTS[experiment]["directory"] / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions for {experiment}: {path}")
    return path


def _canonicalize_counts(counts: dict[str, Counter[str]]) -> dict[str, Counter[str]]:
    canonical_counts: dict[str, Counter[str]] = defaultdict(Counter)
    labels = set(MATRIX_LABELS)
    for reference, row in counts.items():
        reference_label = reference if reference in labels else reference
        for prediction, count in row.items():
            prediction_label = prediction if prediction in labels else prediction
            canonical_counts[reference_label][prediction_label] += count
    for label in MATRIX_LABELS:
        canonical_counts[label]
    return canonical_counts


def _ordered_matrix_labels(counts: dict[str, Counter[str]]) -> list[str]:
    labels = set(MATRIX_LABELS)
    labels.update(counts.keys())
    for row in counts.values():
        labels.update(row.keys())
    return [label for label in MATRIX_LABELS if label in labels] + sorted(label for label in labels if label not in MATRIX_LABELS)


def _row_total(counts: dict[str, Counter[str]], reference: str, labels: list[str]) -> int:
    return sum(counts[reference].get(prediction, 0) for prediction in labels)


def _accuracy(counts: dict[str, Counter[str]], reference: str, labels: list[str]) -> float | None:
    total = _row_total(counts, reference, labels)
    if total == 0:
        return None
    return counts[reference].get(reference, 0) / total


def _deletion_rate(counts: dict[str, Counter[str]], reference: str, labels: list[str]) -> float | None:
    total = _row_total(counts, reference, labels)
    if total == 0:
        return None
    return counts[reference].get(EPSILON, 0) / total


def _group_accuracy(counts: dict[str, Counter[str]], group: list[str], labels: list[str]) -> float | None:
    total = sum(_row_total(counts, vowel, labels) for vowel in group)
    if total == 0:
        return None
    correct = sum(counts[vowel].get(vowel, 0) for vowel in group)
    return correct / total


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
                    *[counts[reference].get(prediction, 0) / total if total else 0.0 for prediction in labels],
                ]
            )


def _hex_color(value: float) -> str:
    value = max(0.0, min(1.0, value))
    start = (248, 250, 252)
    end = (30, 96, 145)
    rgb = tuple(round(start[idx] + (end[idx] - start[idx]) * value) for idx in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _write_heatmap_svg(path: Path, title: str, counts: dict[str, Counter[str]], labels: list[str]) -> None:
    cell = 44
    left = 96
    top = 104
    width = left + cell * len(labels) + 42
    height = top + cell * len(labels) + 74
    cells = []
    for row_idx, reference in enumerate(labels):
        total = _row_total(counts, reference, labels)
        for col_idx, prediction in enumerate(labels):
            value = counts[reference].get(prediction, 0) / total if total else 0.0
            x = left + col_idx * cell
            y = top + row_idx * cell
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{_hex_color(value)}" stroke="#ffffff"/>'
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" text-anchor="middle" class="cell-text">{value:.2f}</text>'
            )

    x_labels = [
        f'<text x="{left + idx * cell + cell / 2}" y="{top - 12}" text-anchor="middle" class="tick">{html.escape(label)}</text>'
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
  <text x="{left + cell * len(labels) / 2}" y="68" text-anchor="middle" class="axis">Predicted vowel</text>
  <text x="24" y="{top + cell * len(labels) / 2}" class="axis" transform="rotate(-90 24 {top + cell * len(labels) / 2})">Reference vowel</text>
  {"".join(x_labels)}
  {"".join(y_labels)}
  {"".join(cells)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def _format_percent(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.2f}"


def _format_points(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.2f}"


def _top_confusions(counts: dict[str, Counter[str]], reference: str, labels: list[str], limit: int = 3) -> list[dict[str, object]]:
    total = _row_total(counts, reference, labels)
    rows = []
    for prediction in labels:
        if prediction == reference:
            continue
        count = counts[reference].get(prediction, 0)
        if count == 0:
            continue
        rows.append(
            {
                "reference": reference,
                "prediction": prediction,
                "count": count,
                "percentage": count / total if total else 0.0,
            }
        )
    rows.sort(key=lambda row: (int(row["count"]), float(row["percentage"]), str(row["prediction"])), reverse=True)
    return rows[:limit]


def _dominant_confusion_label(counts: dict[str, Counter[str]], reference: str, labels: list[str]) -> str:
    top = _top_confusions(counts, reference, labels, limit=1)
    if not top:
        return "none"
    row = top[0]
    return f"{row['reference']}->{row['prediction']} ({int(row['count'])}, {float(row['percentage']) * 100:.2f}%)"


def _interpretation_label(difference: float | None) -> str:
    if difference is None:
        return "No reference tokens"
    if difference <= -0.15:
        return "High degradation"
    if difference <= -0.05:
        return "Moderate degradation"
    if difference < 0:
        return "Low degradation"
    return "No naturalistic degradation"


def _write_accuracy_bar_plot(path: Path, accuracy_rows: list[dict[str, object]]) -> None:
    width = 1100
    height = 640
    left = 78
    top = 72
    plot_width = 960
    plot_height = 400
    group_width = plot_width / len(VOWEL_LABELS)
    bar_width = 13
    experiments = ["E3", "E5", "E8", "E10"]
    colors = {"E3": "#2563eb", "E5": "#16a34a", "E8": "#dc2626", "E10": "#7c3aed"}
    accuracy = {
        (str(row["experiment"]), str(row["vowel"])): None if row["accuracy"] == "" else float(row["accuracy"])
        for row in accuracy_rows
    }

    def y_coord(value: float) -> float:
        return top + plot_height - value * plot_height

    elements = []
    for tick in range(0, 101, 20):
        y = y_coord(tick / 100)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>')
        elements.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick}</text>')

    for group_idx, vowel in enumerate(VOWEL_LABELS):
        group_center = left + group_width * group_idx + group_width / 2
        for exp_idx, experiment in enumerate(experiments):
            value = accuracy.get((experiment, vowel))
            if value is None:
                continue
            x = group_center + (exp_idx - 1.5) * (bar_width + 3) - bar_width / 2
            y = y_coord(value)
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{top + plot_height - y:.1f}" '
                f'fill="{colors[experiment]}" rx="2"/>'
            )
        elements.append(f'<text x="{group_center:.1f}" y="{top + plot_height + 28}" text-anchor="middle" class="tick">{html.escape(vowel)}</text>')

    legend = []
    for idx, experiment in enumerate(experiments):
        x = left + idx * 112
        label = f"{experiment} {EXPERIMENTS[experiment]['test_style']}"
        legend.append(f'<rect x="{x}" y="40" width="14" height="14" fill="{colors[experiment]}" rx="2"/>')
        legend.append(f'<text x="{x + 21}" y="52" class="legend">{html.escape(label)}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <rect width="{width}" height="{height}" fill="#ffffff"/>
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: #1f2933; }}
    .title {{ font-size: 22px; font-weight: 700; }}
    .axis {{ stroke: #243b53; stroke-width: 1.5; }}
    .grid {{ stroke: #d9e2ec; stroke-width: 1; }}
    .tick {{ font-size: 12px; fill: #52606d; }}
    .legend {{ font-size: 12px; font-weight: 700; }}
    .axis-label {{ font-size: 14px; font-weight: 700; }}
  </style>
  <text x="{left}" y="26" class="title">Per-Vowel Accuracy by Style-Controlled Experiment</text>
  {"".join(legend)}
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>
  <text x="22" y="{top + 240}" class="axis-label" transform="rotate(-90 22 {top + 240})">Accuracy (%)</text>
  {"".join(elements)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def _build_accuracy_rows(counts_by_experiment: dict[str, dict[str, Counter[str]]], labels_by_experiment: dict[str, list[str]]) -> list[dict[str, object]]:
    rows = []
    for experiment in ["E3", "E5", "E8", "E10"]:
        counts = counts_by_experiment[experiment]
        labels = labels_by_experiment[experiment]
        for vowel in VOWEL_LABELS:
            total = _row_total(counts, vowel, labels)
            correct = counts[vowel].get(vowel, 0)
            accuracy = correct / total if total else None
            rows.append(
                {
                    "experiment": experiment,
                    "transfer": EXPERIMENTS[experiment]["transfer"],
                    "test_style": EXPERIMENTS[experiment]["test_style"],
                    "vowel": vowel,
                    "reference_count": total,
                    "correct_count": correct,
                    "accuracy": "" if accuracy is None else accuracy,
                    "error_rate": "" if accuracy is None else 1.0 - accuracy,
                }
            )
    return rows


def _build_table1(counts_by_experiment: dict[str, dict[str, Counter[str]]], labels_by_experiment: dict[str, list[str]]) -> list[dict[str, object]]:
    rows = []
    for vowel in VOWEL_LABELS:
        e3 = _accuracy(counts_by_experiment["E3"], vowel, labels_by_experiment["E3"])
        e5 = _accuracy(counts_by_experiment["E5"], vowel, labels_by_experiment["E5"])
        e8 = _accuracy(counts_by_experiment["E8"], vowel, labels_by_experiment["E8"])
        e10 = _accuracy(counts_by_experiment["E10"], vowel, labels_by_experiment["E10"])
        no_transfer_diff = None if e3 is None or e5 is None else e3 - e5
        transfer_diff = None if e8 is None or e10 is None else e8 - e10
        usable_differences = [value for value in [no_transfer_diff, transfer_diff] if value is not None]
        rows.append(
            {
                "vowel": vowel,
                "E3_accuracy": "" if e3 is None else e3,
                "E5_accuracy": "" if e5 is None else e5,
                "E3_minus_E5": "" if no_transfer_diff is None else no_transfer_diff,
                "E8_accuracy": "" if e8 is None else e8,
                "E10_accuracy": "" if e10 is None else e10,
                "E8_minus_E10": "" if transfer_diff is None else transfer_diff,
                "max_absolute_difference": "" if not usable_differences else max(abs(value) for value in usable_differences),
            }
        )
    rows.sort(key=lambda row: float(row["max_absolute_difference"]) if row["max_absolute_difference"] != "" else -1.0, reverse=True)
    return rows


def _build_table2(counts_by_experiment: dict[str, dict[str, Counter[str]]], labels_by_experiment: dict[str, list[str]]) -> list[dict[str, object]]:
    rows = []
    for experiment in ["E3", "E5", "E8", "E10"]:
        for vowel in VOWEL_LABELS:
            for rank, row in enumerate(_top_confusions(counts_by_experiment[experiment], vowel, labels_by_experiment[experiment]), start=1):
                rows.append(
                    {
                        "experiment": experiment,
                        "transfer": EXPERIMENTS[experiment]["transfer"],
                        "test_style": EXPERIMENTS[experiment]["test_style"],
                        "reference": row["reference"],
                        "prediction": row["prediction"],
                        "rank_within_reference": rank,
                        "count": row["count"],
                        "percentage": row["percentage"],
                    }
                )
    return rows


def _build_table3(counts_by_experiment: dict[str, dict[str, Counter[str]]], labels_by_experiment: dict[str, list[str]]) -> list[dict[str, object]]:
    rows = []
    for experiment in ["E3", "E5", "E8", "E10"]:
        oral = _group_accuracy(counts_by_experiment[experiment], ORAL_VOWELS, labels_by_experiment[experiment])
        nasal = _group_accuracy(counts_by_experiment[experiment], NASAL_VOWELS, labels_by_experiment[experiment])
        rows.append(
            {
                "experiment": experiment,
                "transfer": EXPERIMENTS[experiment]["transfer"],
                "test_style": EXPERIMENTS[experiment]["test_style"],
                "oral_accuracy": "" if oral is None else oral,
                "nasal_accuracy": "" if nasal is None else nasal,
                "nasal_minus_oral": "" if oral is None or nasal is None else nasal - oral,
            }
        )
    return rows


def _build_table4(counts_by_experiment: dict[str, dict[str, Counter[str]]], labels_by_experiment: dict[str, list[str]]) -> list[dict[str, object]]:
    rows = []
    for experiment in ["E3", "E5", "E8", "E10"]:
        row = {
            "experiment": experiment,
            "transfer": EXPERIMENTS[experiment]["transfer"],
            "test_style": EXPERIMENTS[experiment]["test_style"],
        }
        for class_name, vowels in VOWEL_CLASSES.items():
            accuracy = _group_accuracy(counts_by_experiment[experiment], vowels, labels_by_experiment[experiment])
            row[f"{class_name}_accuracy"] = "" if accuracy is None else accuracy
        rows.append(row)
    return rows


def _build_table5(counts_by_experiment: dict[str, dict[str, Counter[str]]], labels_by_experiment: dict[str, list[str]]) -> list[dict[str, object]]:
    rows = []
    for vowel in VOWEL_LABELS:
        e3 = _deletion_rate(counts_by_experiment["E3"], vowel, labels_by_experiment["E3"])
        e5 = _deletion_rate(counts_by_experiment["E5"], vowel, labels_by_experiment["E5"])
        e8 = _deletion_rate(counts_by_experiment["E8"], vowel, labels_by_experiment["E8"])
        e10 = _deletion_rate(counts_by_experiment["E10"], vowel, labels_by_experiment["E10"])
        rows.append(
            {
                "vowel": vowel,
                "E3_deletion_rate": "" if e3 is None else e3,
                "E5_deletion_rate": "" if e5 is None else e5,
                "E3_minus_E5": "" if e3 is None or e5 is None else e3 - e5,
                "E8_deletion_rate": "" if e8 is None else e8,
                "E10_deletion_rate": "" if e10 is None else e10,
                "E8_minus_E10": "" if e8 is None or e10 is None else e8 - e10,
            }
        )
    return rows


def _build_table6(counts_by_experiment: dict[str, dict[str, Counter[str]]], labels_by_experiment: dict[str, list[str]]) -> list[dict[str, object]]:
    rows = []
    for pair_name, naturalistic, constrained, transfer_condition in PAIRS:
        pair_rows = []
        for vowel in VOWEL_LABELS:
            naturalistic_accuracy = _accuracy(counts_by_experiment[naturalistic], vowel, labels_by_experiment[naturalistic])
            constrained_accuracy = _accuracy(counts_by_experiment[constrained], vowel, labels_by_experiment[constrained])
            difference = None if naturalistic_accuracy is None or constrained_accuracy is None else naturalistic_accuracy - constrained_accuracy
            pair_rows.append(
                {
                    "pair": pair_name,
                    "transfer_condition": transfer_condition,
                    "vowel": vowel,
                    "naturalistic_experiment": naturalistic,
                    "constrained_experiment": constrained,
                    "naturalistic_accuracy": "" if naturalistic_accuracy is None else naturalistic_accuracy,
                    "constrained_accuracy": "" if constrained_accuracy is None else constrained_accuracy,
                    "accuracy_difference_naturalistic_minus_constrained": "" if difference is None else difference,
                    "dominant_naturalistic_confusion": _dominant_confusion_label(
                        counts_by_experiment[naturalistic],
                        vowel,
                        labels_by_experiment[naturalistic],
                    ),
                    "interpretation_label": _interpretation_label(difference),
                }
            )
        pair_rows.sort(
            key=lambda row: float(row["accuracy_difference_naturalistic_minus_constrained"])
            if row["accuracy_difference_naturalistic_minus_constrained"] != ""
            else 1.0
        )
        for rank, row in enumerate(pair_rows, start=1):
            row["rank"] = rank
            rows.append({"rank": row.pop("rank"), **row})
    return rows


def _write_markdown_summary(
    path: Path,
    table1_rows: list[dict[str, object]],
    table2_rows: list[dict[str, object]],
    table3_rows: list[dict[str, object]],
    table5_rows: list[dict[str, object]],
    table6_rows: list[dict[str, object]],
) -> None:
    top_no_transfer = [row for row in table6_rows if row["pair"] == "E3_vs_E5"][:5]
    top_transfer = [row for row in table6_rows if row["pair"] == "E8_vs_E10"][:5]
    oral_lines = [
        f"- {row['experiment']}: oral {_format_percent(float(row['oral_accuracy']) if row['oral_accuracy'] != '' else None)}%, nasal {_format_percent(float(row['nasal_accuracy']) if row['nasal_accuracy'] != '' else None)}%, nasal-minus-oral {_format_points(float(row['nasal_minus_oral']) if row['nasal_minus_oral'] != '' else None)} pp"
        for row in table3_rows
    ]
    deletion_rank = sorted(
        table5_rows,
        key=lambda row: max(
            [float(value) for value in [row["E3_minus_E5"], row["E8_minus_E10"]] if value != ""],
            default=-999.0,
        ),
        reverse=True,
    )[:5]
    confusion_rank = sorted(table2_rows, key=lambda row: int(row["count"]), reverse=True)[:10]

    lines = [
        "# Style-Controlled Vowel Analysis",
        "",
        "Only E3, E5, E8, and E10 are included. E3/E5 hold mixed training constant without transfer; E8/E10 hold mixed training constant with Yoruba transfer. The varying factor within each pair is evaluation speech style.",
        "",
        "## Overall Findings",
        "",
        "The strongest style-sensitive vowels are those with the most negative naturalistic-minus-constrained accuracy differences in Table 6. Table 1 ranks vowels by the largest absolute style difference observed across the two controlled pairs.",
        "",
        "## Largest Style-Sensitive Vowels",
        "",
        "No transfer, E3 minus E5:",
        *[
            f"- {row['vowel']}: {float(row['accuracy_difference_naturalistic_minus_constrained']) * 100:.2f} pp; dominant confusion {row['dominant_naturalistic_confusion']}; {row['interpretation_label']}"
            for row in top_no_transfer
            if row["accuracy_difference_naturalistic_minus_constrained"] != ""
        ],
        "",
        "Transfer, E8 minus E10:",
        *[
            f"- {row['vowel']}: {float(row['accuracy_difference_naturalistic_minus_constrained']) * 100:.2f} pp; dominant confusion {row['dominant_naturalistic_confusion']}; {row['interpretation_label']}"
            for row in top_transfer
            if row["accuracy_difference_naturalistic_minus_constrained"] != ""
        ],
        "",
        "## Major Confusion Pathways",
        "",
        *[
            f"- {row['experiment']} {row['reference']}->{row['prediction']}: {row['count']} ({float(row['percentage']) * 100:.2f}%)"
            for row in confusion_rank
        ],
        "",
        "## Oral vs Nasal Comparison",
        "",
        *oral_lines,
        "",
        "## Deletion Patterns",
        "",
        *[
            f"- {row['vowel']}: E3-E5 {float(row['E3_minus_E5']) * 100:.2f} pp; E8-E10 {float(row['E8_minus_E10']) * 100:.2f} pp"
            for row in deletion_rank
            if row["E3_minus_E5"] != "" and row["E8_minus_E10"] != ""
        ],
        "",
        "## Output Files",
        "",
        "- `table1_vowel_accuracy.csv`",
        "- `table2_main_vowel_confusions.csv`",
        "- `table3_oral_vs_nasal.csv`",
        "- `table4_vowel_classes.csv`",
        "- `table5_deletion_rates.csv`",
        "- `table6_style_sensitivity.csv`",
        "- Per-experiment matrices and row-normalized heatmaps under `E3/`, `E5/`, `E8/`, and `E10/`",
        "- `per_vowel_accuracy_by_experiment.svg`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create style-controlled vowel analysis for the ASR paper.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="results/style_controlled_vowel_analysis")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    counts_by_experiment: dict[str, dict[str, Counter[str]]] = {}
    labels_by_experiment: dict[str, list[str]] = {}
    manifest = []

    for experiment in ["E3", "E5", "E8", "E10"]:
        predictions_csv = _find_predictions(results_root, experiment)
        rows = _read_predictions(predictions_csv)
        analysis = _analyze_rows(rows)
        counts = _canonicalize_counts(analysis["vowel_counts"])
        labels = _ordered_matrix_labels(counts)
        counts_by_experiment[experiment] = counts
        labels_by_experiment[experiment] = labels

        experiment_dir = output_dir / experiment
        experiment_dir.mkdir(parents=True, exist_ok=True)
        _write_count_matrix(experiment_dir / "vowel_confusion_matrix.csv", counts, labels)
        _write_row_normalized_matrix(experiment_dir / "vowel_confusion_matrix_row_normalized.csv", counts, labels)
        _write_heatmap_svg(
            experiment_dir / "vowel_confusion_heatmap_row_normalized.svg",
            f"{experiment} Row-Normalized Vowel Confusion",
            counts,
            labels,
        )
        manifest.append({"experiment": experiment, "predictions_csv": str(predictions_csv), **EXPERIMENTS[experiment]})

    accuracy_rows = _build_accuracy_rows(counts_by_experiment, labels_by_experiment)
    table1_rows = _build_table1(counts_by_experiment, labels_by_experiment)
    table2_rows = _build_table2(counts_by_experiment, labels_by_experiment)
    table3_rows = _build_table3(counts_by_experiment, labels_by_experiment)
    table4_rows = _build_table4(counts_by_experiment, labels_by_experiment)
    table5_rows = _build_table5(counts_by_experiment, labels_by_experiment)
    table6_rows = _build_table6(counts_by_experiment, labels_by_experiment)

    _write_table(output_dir / "per_experiment_vowel_accuracy.csv", accuracy_rows)
    _write_table(output_dir / "table1_vowel_accuracy.csv", table1_rows)
    _write_table(output_dir / "table2_main_vowel_confusions.csv", table2_rows)
    _write_table(output_dir / "table3_oral_vs_nasal.csv", table3_rows)
    _write_table(output_dir / "table4_vowel_classes.csv", table4_rows)
    _write_table(output_dir / "table5_deletion_rates.csv", table5_rows)
    _write_table(output_dir / "table6_style_sensitivity.csv", table6_rows)
    _write_accuracy_bar_plot(output_dir / "per_vowel_accuracy_by_experiment.svg", accuracy_rows)
    _write_markdown_summary(output_dir / "summary.md", table1_rows, table2_rows, table3_rows, table5_rows, table6_rows)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved style-controlled vowel analysis to: {output_dir}")


if __name__ == "__main__":
    main()
