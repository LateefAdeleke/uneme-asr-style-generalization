#!/usr/bin/env python3
from __future__ import annotations

"""Style-controlled tone analysis for mixed-training ASR experiments.

This report isolates the experiments needed to test whether tone recognition
degrades on naturalistic speech when training style is held constant:

- E3: mixed training -> naturalistic test, no transfer
- E5: mixed training -> constrained test, no transfer
- E8: transfer + mixed training -> naturalistic test
- E10: transfer + mixed training -> constrained test
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_tone_vowel_confusions import EPSILON, _analyze_rows, _read_predictions
from style_aware_tone_analysis import (
    _canonicalize_counts,
    _write_heatmap_svg,
    _write_count_matrix,
    _write_row_normalized_matrix,
)


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

TONE_LABELS = ["high", "low", "mid", "low+high", EPSILON]
HEATMAP_TONE_LABELS = ["high", "low", "mid", EPSILON]
REPORT_TONES = ["high", "low", "mid"]
PATHWAYS = [
    ("mid", "high"),
    ("mid", "low"),
    ("mid", EPSILON),
    ("high", "mid"),
    ("low", "mid"),
]


def _find_predictions(results_root: Path, experiment: str) -> Path:
    path = results_root / EXPERIMENTS[experiment]["directory"] / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions for {experiment}: {path}")
    return path


def _row_total(counts: dict[str, Counter[str]], reference: str) -> int:
    return sum(counts[reference].get(prediction, 0) for prediction in TONE_LABELS)


def _overall_tone_error_rate(counts: dict[str, Counter[str]]) -> float:
    total = 0
    correct = 0
    for tone in TONE_LABELS:
        if tone == EPSILON:
            continue
        total += _row_total(counts, tone)
        correct += counts[tone].get(tone, 0)
    return 1.0 - (correct / total if total else 0.0)


def _accuracy(counts: dict[str, Counter[str]], tone: str) -> float:
    total = _row_total(counts, tone)
    return counts[tone].get(tone, 0) / total if total else 0.0


def _row_pct(counts: dict[str, Counter[str]], reference: str, prediction: str) -> float:
    total = _row_total(counts, reference)
    return counts[reference].get(prediction, 0) / total if total else 0.0


def _write_pathway_table(path: Path, experiment_rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "experiment",
        "transfer",
        "train_style",
        "test_style",
        "reference_tone",
        "predicted_tone",
        "count",
        "row_percent",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(experiment_rows)


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_accuracy_bar_plot(path: Path, table_rows: list[dict[str, object]]) -> None:
    import html

    width = 920
    height = 560
    left = 90
    top = 70
    plot_width = 760
    plot_height = 350
    group_width = plot_width / len(table_rows)
    bar_width = 34
    colors = {"high": "#2f80ed", "low": "#d64545", "mid": "#2f9e44"}

    def y_coord(percent: float) -> float:
        return top + plot_height - (percent / 100.0 * plot_height)

    elements = []
    for tick in range(0, 101, 20):
        y = y_coord(tick)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>')
        elements.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick}</text>')

    for group_idx, row in enumerate(table_rows):
        group_center = left + group_width * group_idx + group_width / 2
        for tone_idx, tone in enumerate(REPORT_TONES):
            value = float(row[f"{tone}_accuracy"]) * 100.0
            x = group_center + (tone_idx - 1) * (bar_width + 8) - bar_width / 2
            y = y_coord(value)
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{top + plot_height - y:.1f}" '
                f'fill="{colors[tone]}" rx="2"/>'
            )
            elements.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="bar-label">{value:.1f}</text>')

        label_lines = [
            html.escape(str(row["experiment"])),
            html.escape(str(row["test_style"])),
            f"transfer={html.escape(str(row['transfer']).lower())}",
        ]
        for line_idx, label in enumerate(label_lines):
            elements.append(
                f'<text x="{group_center:.1f}" y="{top + plot_height + 28 + line_idx * 16}" text-anchor="middle" class="tick">{label}</text>'
            )

    legend = []
    for idx, tone in enumerate(REPORT_TONES):
        x = left + idx * 115
        legend.append(f'<rect x="{x}" y="38" width="14" height="14" fill="{colors[tone]}" rx="2"/>')
        legend.append(f'<text x="{x + 22}" y="50" class="legend">{tone.title()}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <rect width="{width}" height="{height}" fill="#ffffff"/>
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: #1f2933; }}
    .title {{ font-size: 22px; font-weight: 700; }}
    .axis {{ stroke: #243b53; stroke-width: 1.5; }}
    .grid {{ stroke: #d9e2ec; stroke-width: 1; }}
    .tick {{ font-size: 12px; fill: #52606d; }}
    .legend {{ font-size: 13px; font-weight: 700; }}
    .axis-label {{ font-size: 14px; font-weight: 700; }}
    .bar-label {{ font-size: 11px; font-weight: 700; }}
  </style>
  <text x="{left}" y="26" class="title">Tone Accuracy by Style-Controlled Experiment</text>
  {"".join(legend)}
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>
  <text x="24" y="{top + 220}" class="axis-label" transform="rotate(-90 24 {top + 220})">Accuracy (%)</text>
  {"".join(elements)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}"


def _write_markdown_summary(
    path: Path,
    table1_rows: list[dict[str, object]],
    table2_rows: list[dict[str, object]],
    table3_rows: list[dict[str, object]],
) -> None:
    table1_by_exp = {str(row["experiment"]): row for row in table1_rows}
    mid_by_exp = {str(row["experiment"]): row for row in table2_rows}

    e3 = table1_by_exp["E3"]
    e5 = table1_by_exp["E5"]
    e8 = table1_by_exp["E8"]
    e10 = table1_by_exp["E10"]

    no_transfer_gap = float(e3["overall_tone_error_rate"]) - float(e5["overall_tone_error_rate"])
    transfer_gap = float(e8["overall_tone_error_rate"]) - float(e10["overall_tone_error_rate"])

    lines = [
        "# Style-Controlled Tone Analysis",
        "",
        "Only E3, E5, E8, and E10 are included. All four use mixed training; the controlled comparison is naturalistic versus constrained test speech, once without transfer and once with cross-lingual transfer.",
        "",
        "## Key numerical findings",
        "",
        f"- Without transfer, overall tone error is {_format_pct(float(e3['overall_tone_error_rate']))}% on E3 naturalistic speech versus {_format_pct(float(e5['overall_tone_error_rate']))}% on E5 constrained speech, a naturalistic increase of {_format_pct(no_transfer_gap)} percentage points.",
        f"- With transfer, overall tone error is {_format_pct(float(e8['overall_tone_error_rate']))}% on E8 naturalistic speech versus {_format_pct(float(e10['overall_tone_error_rate']))}% on E10 constrained speech, a naturalistic increase of {_format_pct(transfer_gap)} percentage points.",
        f"- The largest no-transfer tone accuracy drop is {max((row for row in table3_rows if row['pair'] == 'E3_vs_E5'), key=lambda row: abs(float(row['difference_naturalistic_minus_constrained'])) )['tone']}.",
        f"- The largest transfer tone accuracy drop is {max((row for row in table3_rows if row['pair'] == 'E8_vs_E10'), key=lambda row: abs(float(row['difference_naturalistic_minus_constrained'])) )['tone']}.",
        f"- MID is especially informative: E3 MID correct is {float(mid_by_exp['E3']['mid_correct_percent']):.2f}%, with MID->HIGH {float(mid_by_exp['E3']['mid_to_high_percent']):.2f}%, MID->LOW {float(mid_by_exp['E3']['mid_to_low_percent']):.2f}%, and MID-><eps> {float(mid_by_exp['E3']['mid_to_eps_percent']):.2f}%. E5 MID correct is {float(mid_by_exp['E5']['mid_correct_percent']):.2f}%.",
        f"- Under transfer, E8 MID correct is {float(mid_by_exp['E8']['mid_correct_percent']):.2f}% and E10 MID correct is {float(mid_by_exp['E10']['mid_correct_percent']):.2f}%.",
        "",
        "## Claim",
        "",
        "Holding training style constant, tone recognition degrades under naturalistic speech. The degradation persists under cross-lingual transfer, although the size and affected tone categories should be read from the pairwise table because the transfer condition changes the distribution of errors.",
        "",
        "## Output files",
        "",
        "- `table1_experiment_tone_accuracy.csv`",
        "- `table2_mid_confusion_pathways.csv`",
        "- `table3_pairwise_tone_accuracy.csv`",
        "- `main_off_diagonal_pathways.csv`",
        "- Per-experiment confusion matrices and row-normalized matrices under `E3/`, `E5/`, `E8/`, and `E10/`",
        "- Row-normalized heatmaps under `heatmaps/`",
        "- Combined bar plot: `tone_accuracy_by_experiment.svg`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create style-controlled tone analysis for the ASR paper.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="results/style_controlled_tone_analysis")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    heatmap_dir = output_dir / "heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    table1_rows: list[dict[str, object]] = []
    table2_rows: list[dict[str, object]] = []
    table3_rows: list[dict[str, object]] = []
    pathway_rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    counts_by_experiment: dict[str, dict[str, Counter[str]]] = {}

    for experiment in ["E3", "E5", "E8", "E10"]:
        metadata = EXPERIMENTS[experiment]
        predictions_csv = _find_predictions(results_root, experiment)
        rows = _read_predictions(predictions_csv)
        analysis = _analyze_rows(rows)
        counts = _canonicalize_counts(analysis["tone_counts"])
        counts_by_experiment[experiment] = counts

        experiment_dir = output_dir / experiment
        experiment_dir.mkdir(parents=True, exist_ok=True)
        _write_count_matrix(experiment_dir / "tone_confusion_matrix.csv", counts, TONE_LABELS)
        _write_row_normalized_matrix(experiment_dir / "tone_confusion_matrix_row_normalized.csv", counts, TONE_LABELS)
        _write_heatmap_svg(
            heatmap_dir / f"{experiment}_tone_confusion_row_normalized.svg",
            f"{experiment} Row-Normalized Tone Confusion",
            counts,
            HEATMAP_TONE_LABELS,
            normalized=True,
        )

        table1_rows.append(
            {
                "experiment": experiment,
                "transfer": metadata["transfer"],
                "train_style": metadata["train_style"],
                "test_style": metadata["test_style"],
                "high_accuracy": _accuracy(counts, "high"),
                "low_accuracy": _accuracy(counts, "low"),
                "mid_accuracy": _accuracy(counts, "mid"),
                "overall_tone_error_rate": _overall_tone_error_rate(counts),
            }
        )
        table2_rows.append(
            {
                "experiment": experiment,
                "test_style": metadata["test_style"],
                "mid_to_high_percent": _row_pct(counts, "mid", "high") * 100.0,
                "mid_to_low_percent": _row_pct(counts, "mid", "low") * 100.0,
                "mid_to_eps_percent": _row_pct(counts, "mid", EPSILON) * 100.0,
                "mid_correct_percent": _row_pct(counts, "mid", "mid") * 100.0,
            }
        )
        for reference, prediction in PATHWAYS:
            pathway_rows.append(
                {
                    "experiment": experiment,
                    "transfer": metadata["transfer"],
                    "train_style": metadata["train_style"],
                    "test_style": metadata["test_style"],
                    "reference_tone": reference,
                    "predicted_tone": prediction,
                    "count": counts[reference].get(prediction, 0),
                    "row_percent": _row_pct(counts, reference, prediction) * 100.0,
                }
            )
        manifest.append({"experiment": experiment, "predictions_csv": str(predictions_csv), **metadata})

    for pair_name, naturalistic, constrained in [("E3_vs_E5", "E3", "E5"), ("E8_vs_E10", "E8", "E10")]:
        for tone in REPORT_TONES:
            naturalistic_accuracy = _accuracy(counts_by_experiment[naturalistic], tone)
            constrained_accuracy = _accuracy(counts_by_experiment[constrained], tone)
            table3_rows.append(
                {
                    "pair": pair_name,
                    "tone": tone,
                    "naturalistic_experiment": naturalistic,
                    "constrained_experiment": constrained,
                    "naturalistic_accuracy": naturalistic_accuracy,
                    "constrained_accuracy": constrained_accuracy,
                    "difference_naturalistic_minus_constrained": naturalistic_accuracy - constrained_accuracy,
                }
            )

    _write_table(output_dir / "table1_experiment_tone_accuracy.csv", table1_rows)
    _write_table(output_dir / "table2_mid_confusion_pathways.csv", table2_rows)
    _write_table(output_dir / "table3_pairwise_tone_accuracy.csv", table3_rows)
    _write_pathway_table(output_dir / "main_off_diagonal_pathways.csv", pathway_rows)
    _write_accuracy_bar_plot(output_dir / "tone_accuracy_by_experiment.svg", table1_rows)
    _write_markdown_summary(output_dir / "summary.md", table1_rows, table2_rows, table3_rows)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved style-controlled tone analysis to: {output_dir}")


if __name__ == "__main__":
    main()
