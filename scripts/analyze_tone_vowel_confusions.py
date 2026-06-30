#!/usr/bin/env python3
from __future__ import annotations

"""Post hoc vowel and tone confusion analysis from saved ASR predictions.

This script reads a `predictions.csv` file produced by the experiment pipeline,
aligns the `reference` and `prediction` strings, and computes:

- a vowel confusion matrix
- a tone confusion matrix
- a summary JSON including tone error rate

Tone is modeled on vowel graphemes, not as a separate token stream. For the
current language assumptions:

- grave accent = low tone
- acute accent = high tone
- unmarked vowel = mid tone

The analysis therefore first segments text into Unicode grapheme-like vowel
units, then derives vowel identity and tone category from each vowel grapheme.

The CLI supports:

- single-file analysis for one `predictions.csv`
- batch analysis across all experiment outputs under a results root
"""

import argparse
import csv
import html
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


TONE_MARKS = {
    "\u0300": "low",
    "\u0301": "high",
}
VOWEL_INVENTORY = {"a", "e", "ẹ", "i", "o", "ọ", "u", "an", "ẹn", "ọn", "un", "in"}
VOWEL_LABEL_ORDER = [EPSILON := "<eps>", "a", "e", "ẹ", "i", "o", "ọ", "u", "an", "ẹn", "ọn", "un", "in"]
TONE_LABEL_ORDER = [EPSILON, "high", "low", "mid"]
BASE_ORAL_VOWELS = {"a", "e", "ẹ", "i", "o", "ọ", "u"}


def _read_predictions(path: Path) -> list[dict[str, str]]:
    """Load experiment predictions from a CSV exported by the ASR pipeline."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load rows from a UTF-8 CSV file."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _select_text_column(rows: Sequence[dict[str, str]], preferred: str | None = None) -> str:
    """Choose the transcript column to use for distribution counts."""

    if not rows:
        raise ValueError("Cannot select a text column from an empty CSV.")
    candidates = [
        preferred,
        "transcription_norm",
        "transcription",
        "reference",
        "text",
    ]
    for candidate in candidates:
        if candidate and candidate in rows[0]:
            return candidate
    available = ", ".join(rows[0].keys())
    raise KeyError(f"No transcript column found. Available columns: {available}")


def _split_graphemes(text: str) -> list[str]:
    """Split text into simple grapheme-like units using Unicode decomposition.

    Base characters start a new unit and following combining marks are attached
    to that unit. This lets accented vowels stay together as one analysis unit.
    """

    normalized = unicodedata.normalize("NFD", text)
    graphemes: list[str] = []
    current = ""
    for char in normalized:
        if unicodedata.combining(char):
            if current:
                current += char
            continue
        if current:
            graphemes.append(unicodedata.normalize("NFC", current))
        current = char
    if current:
        graphemes.append(unicodedata.normalize("NFC", current))
    return graphemes


def _is_vowel_grapheme(grapheme: str) -> bool:
    """Return True when the grapheme's base character is treated as an oral vowel."""

    if not grapheme:
        return False
    normalized = unicodedata.normalize("NFD", grapheme)
    base = normalized[0].lower()
    base_with_nontone_marks = _vowel_identity(grapheme)
    return base_with_nontone_marks in BASE_ORAL_VOWELS


def _vowel_identity(grapheme: str) -> str:
    """Return the vowel identity with tone marks stripped off.

    This preserves segment identity while discarding the tone layer so vowel
    confusions and tone confusions can be analyzed separately.
    """

    normalized = unicodedata.normalize("NFD", grapheme)
    chars = [char for char in normalized if char not in TONE_MARKS]
    return unicodedata.normalize("NFC", "".join(chars)).lower()


def _tone_label(grapheme: str) -> str:
    """Return the tone category carried by a vowel grapheme.

    Assumptions:
    - acute accent => high
    - grave accent => low
    - no tone mark => mid
    """

    normalized = unicodedata.normalize("NFD", grapheme)
    labels = [TONE_MARKS[char] for char in normalized[1:] if char in TONE_MARKS]
    if not labels:
        return "mid"
    return "+".join(labels)


def _extract_vowel_units(text: str) -> list[dict[str, str]]:
    """Extract vowel units using the exact language vowel inventory.

    Inventory:
    - oral vowels: a, e, ẹ, i, o, ọ, u
    - nasal vowels: an, ẹn, ọn, un, in

    Tone is carried by the vowel portion of the unit:
    - acute => high
    - grave => low
    - unmarked => mid
    """

    graphemes = _split_graphemes(text)
    units: list[dict[str, str]] = []
    idx = 0
    while idx < len(graphemes):
        grapheme = graphemes[idx]
        if not _is_vowel_grapheme(grapheme):
            idx += 1
            continue

        vowel = _vowel_identity(grapheme)
        surface = grapheme
        if idx + 1 < len(graphemes) and graphemes[idx + 1].lower() == "n":
            candidate = f"{vowel}n"
            if candidate in VOWEL_INVENTORY:
                vowel = candidate
                surface += graphemes[idx + 1]
                idx += 1

        if vowel in VOWEL_INVENTORY:
            units.append(
                {
                    "surface": surface,
                    "vowel": vowel,
                    "tone": _tone_label(grapheme),
                }
            )
        idx += 1
    return units


def _align_sequences(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[str | None, str | None]]:
    """Align two grapheme sequences with Levenshtein backtracking.

    Returns pairs of `(reference_token, hypothesis_token)` where either side can
    be `None` to represent an insertion/deletion.
    """

    m = len(ref)
    n = len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            substitution_cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + substitution_cost,
            )

    alignment: list[tuple[str | None, str | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            substitution_cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + substitution_cost:
                alignment.append((ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((ref[i - 1], None))
            i -= 1
            continue
        alignment.append((None, hyp[j - 1]))
        j -= 1

    alignment.reverse()
    return alignment


def _write_matrix(path: Path, row_labels: list[str], col_labels: list[str], counts: dict[str, Counter[str]]) -> None:
    """Write a confusion matrix CSV from nested reference/prediction counts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference\\prediction", *col_labels])
        for row_label in row_labels:
            writer.writerow([row_label, *[counts[row_label].get(col_label, 0) for col_label in col_labels]])


def _write_heatmap(
    path: Path,
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    counts: dict[str, Counter[str]],
) -> bool:
    """Render and save a confusion-matrix heatmap PNG.

    Returns True when the heatmap was saved and False when matplotlib is not
    installed.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - dependency-driven
        return False

    matrix = [[counts[row_label].get(col_label, 0) for col_label in col_labels] for row_label in row_labels]
    path.parent.mkdir(parents=True, exist_ok=True)

    fig_width = max(6, len(col_labels) * 1.2)
    fig_height = max(5, len(row_labels) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(col_labels):
            ax.text(col_idx, row_idx, str(counts[row_label].get(col_label, 0)), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return True


def _rank_values(values: Sequence[float]) -> list[float]:
    """Return average ranks for values, using 1-based ranks."""

    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        average_rank = (idx + 1 + end) / 2.0
        for original_idx, _ in indexed[idx:end]:
            ranks[original_idx] = average_rank
        idx = end
    return ranks


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Compute Pearson correlation without requiring scipy."""

    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denominator_x == 0 or denominator_y == 0:
        return None
    return numerator / (denominator_x * denominator_y)


def _spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Compute Spearman rank correlation without requiring scipy."""

    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson_correlation(_rank_values(xs), _rank_values(ys))


def _distribution_error_rows(
    counts: dict[str, Counter[str]],
    labels: list[str],
    label_type: str,
) -> list[dict[str, object]]:
    """Create per-label frequency and error-rate rows from a confusion matrix."""

    reference_total = sum(
        sum(prediction_counts.values())
        for reference_label, prediction_counts in counts.items()
        if reference_label != EPSILON
    )
    rows: list[dict[str, object]] = []
    for label in labels:
        if label == EPSILON:
            continue
        total = sum(counts[label].values())
        if total == 0:
            continue
        correct = counts[label].get(label, 0)
        errors = total - correct
        rows.append(
            {
                "label_type": label_type,
                "label": label,
                "reference_count": total,
                "reference_proportion": float(total) / max(reference_total, 1),
                "correct_count": correct,
                "error_count": errors,
                "error_rate": float(errors) / total,
            }
        )
    return rows


def _correlation_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize whether distribution size corresponds to error rate."""

    counts = [float(row["reference_count"]) for row in rows]
    error_rates = [float(row["error_rate"]) for row in rows]
    log_counts = [math.log10(count) for count in counts if count > 0]
    filtered_error_rates = [
        float(row["error_rate"])
        for row in rows
        if float(row["reference_count"]) > 0
    ]
    return {
        "n_labels": len(rows),
        "pearson_count_vs_error_rate": _pearson_correlation(counts, error_rates),
        "pearson_log_count_vs_error_rate": _pearson_correlation(log_counts, filtered_error_rates),
        "spearman_count_vs_error_rate": _spearman_correlation(counts, error_rates),
        "interpretation_hint": "Negative values mean more frequent reference labels tend to have lower error rates.",
    }


def _training_correlation_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize whether training exposure corresponds to test error rate."""

    usable_rows = [row for row in rows if float(row["training_count"]) > 0]
    counts = [float(row["training_count"]) for row in usable_rows]
    error_rates = [float(row["error_rate"]) for row in usable_rows]
    log_counts = [math.log10(count) for count in counts]
    return {
        "n_labels": len(usable_rows),
        "pearson_training_count_vs_error_rate": _pearson_correlation(counts, error_rates),
        "pearson_log_training_count_vs_error_rate": _pearson_correlation(log_counts, error_rates),
        "spearman_training_count_vs_error_rate": _spearman_correlation(counts, error_rates),
        "interpretation_hint": "Negative values mean labels seen more often during training tend to have lower test error rates.",
    }


def _write_distribution_error_analysis(
    output_dir: Path,
    vowel_rows: list[dict[str, object]],
    tone_rows: list[dict[str, object]],
) -> None:
    """Write per-label distribution/error rows and correlation summaries."""

    rows = [*vowel_rows, *tone_rows]
    csv_path = output_dir / "distribution_error_analysis.csv"
    fieldnames = [
        "label_type",
        "label",
        "reference_count",
        "reference_proportion",
        "correct_count",
        "error_count",
        "error_rate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "vowel": _correlation_summary(vowel_rows),
        "tone": _correlation_summary(tone_rows),
    }
    summary_path = output_dir / "distribution_error_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved distribution/error table to: {csv_path}")
    print(f"Saved distribution/error summary to: {summary_path}")


def _count_units_by_label(rows: Iterable[dict[str, str]], text_column: str) -> dict[str, Counter[str]]:
    """Count vowel and tone labels in a transcript column."""

    counts = {
        "vowel": Counter(),
        "tone": Counter(),
    }
    for row in rows:
        for unit in _extract_vowel_units(row.get(text_column, "")):
            counts["vowel"][unit["vowel"]] += 1
            counts["tone"][unit["tone"]] += 1
    return counts


def _training_distribution_error_rows(
    distribution_rows: list[dict[str, object]],
    training_counts: dict[str, Counter[str]],
    label_type: str,
) -> list[dict[str, object]]:
    """Join training distribution counts with held-out error rates."""

    total_training_units = sum(training_counts[label_type].values())
    rows: list[dict[str, object]] = []
    for row in distribution_rows:
        label = str(row["label"])
        training_count = training_counts[label_type].get(label, 0)
        rows.append(
            {
                "label_type": label_type,
                "label": label,
                "training_count": training_count,
                "training_proportion": float(training_count) / max(total_training_units, 1),
                "test_reference_count": row["reference_count"],
                "test_error_count": row["error_count"],
                "error_rate": row["error_rate"],
            }
        )
    return rows


def _write_training_distribution_error_analysis(
    output_dir: Path,
    training_manifest: Path,
    text_column: str,
    vowel_distribution_rows: list[dict[str, object]],
    tone_distribution_rows: list[dict[str, object]],
) -> None:
    """Write training-distribution versus held-out error analysis."""

    training_rows = _read_csv_rows(training_manifest)
    training_counts = _count_units_by_label(training_rows, text_column)
    vowel_rows = _training_distribution_error_rows(vowel_distribution_rows, training_counts, "vowel")
    tone_rows = _training_distribution_error_rows(tone_distribution_rows, training_counts, "tone")
    rows = [*vowel_rows, *tone_rows]

    csv_path = output_dir / "training_distribution_error_analysis.csv"
    fieldnames = [
        "label_type",
        "label",
        "training_count",
        "training_proportion",
        "test_reference_count",
        "test_error_count",
        "error_rate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "training_manifest": str(training_manifest),
        "text_column": text_column,
        "vowel": _training_correlation_summary(vowel_rows),
        "tone": _training_correlation_summary(tone_rows),
    }
    summary_path = output_dir / "training_distribution_error_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved training distribution/error table to: {csv_path}")
    print(f"Saved training distribution/error summary to: {summary_path}")
    _write_training_distribution_svg(output_dir, vowel_rows, "vowel")
    _write_training_distribution_svg(output_dir, tone_rows, "tone")


def _write_training_distribution_svg(output_dir: Path, rows: list[dict[str, object]], label_type: str) -> None:
    """Write a self-contained SVG scatter plot for training count vs error rate."""

    plot_rows = [row for row in rows if float(row["training_count"]) > 0]
    if len(plot_rows) < 2:
        return

    xs = [math.log10(float(row["training_count"])) for row in plot_rows]
    ys = [float(row["error_rate"]) for row in plot_rows]
    min_x = min(xs)
    max_x = max(xs)
    max_y = max(0.5 if label_type == "tone" else 0.85, max(ys) * 1.08)

    left = 90
    top = 50
    width = 640
    height = 390

    def x_coord(value: float) -> float:
        return left + (value - min_x) / max(max_x - min_x, 1e-9) * width

    def y_coord(value: float) -> float:
        return top + (max_y - value) / max(max_y, 1e-9) * height

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator if denominator else 0.0
    intercept = mean_y - slope * mean_x
    line_y1 = intercept + slope * min_x
    line_y2 = intercept + slope * max_x
    log_correlation = _pearson_correlation(xs, ys)
    spearman = _spearman_correlation([float(row["training_count"]) for row in plot_rows], ys)
    experiment_name = output_dir.parent.name
    y_axis_label = f"Held-out {label_type} error rate"

    x_ticks = [
        min(float(row["training_count"]) for row in plot_rows),
        sorted(float(row["training_count"]) for row in plot_rows)[len(plot_rows) // 2],
        max(float(row["training_count"]) for row in plot_rows),
    ]
    x_tick_elements = []
    for tick in x_ticks:
        x_pos = x_coord(math.log10(tick))
        x_tick_elements.append(f'<line x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" y2="{top + height}" class="grid"/>')
        x_tick_elements.append(f'<text x="{x_pos - 18:.1f}" y="462" class="tick">{tick:,.0f}</text>')

    y_step = 0.1 if max_y <= 0.6 else 0.2
    y_tick_values = []
    tick = 0.0
    while tick <= max_y + 1e-9:
        y_tick_values.append(tick)
        tick += y_step
    y_tick_elements = []
    for tick in y_tick_values:
        y_pos = y_coord(tick)
        y_tick_elements.append(f'<line x1="{left}" y1="{y_pos:.1f}" x2="{left + width}" y2="{y_pos:.1f}" class="grid"/>')
        y_tick_elements.append(f'<text x="54" y="{y_pos + 4:.1f}" class="tick">{tick:.1f}</text>')

    point_elements = []
    for row, x, y in zip(plot_rows, xs, ys):
        label = html.escape(str(row["label"]))
        x_pos = x_coord(x)
        y_pos = y_coord(y)
        css_class = "point accent" if y >= 0.5 else "point"
        text_x = min(x_pos + 10, left + width + 18)
        point_elements.append(f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="7" class="{css_class}"/>')
        point_elements.append(f'<text x="{text_x:.1f}" y="{y_pos + 4:.1f}" class="label">{label}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="560" viewBox="0 0 900 560" role="img" aria-labelledby="title desc">
  <title id="title">{html.escape(experiment_name)} {label_type} training distribution versus held-out error rate</title>
  <desc id="desc">Scatter plot showing training {label_type} counts on a log scale against held-out {label_type} error rate.</desc>
  <rect width="900" height="560" fill="#ffffff"/>
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: #1f2933; }}
    .title {{ font-size: 22px; font-weight: 700; }}
    .subtitle {{ font-size: 13px; fill: #52606d; }}
    .axis {{ stroke: #243b53; stroke-width: 1.5; }}
    .grid {{ stroke: #d9e2ec; stroke-width: 1; }}
    .tick {{ font-size: 12px; fill: #52606d; }}
    .label {{ font-size: 13px; font-weight: 700; }}
    .axis-label {{ font-size: 14px; font-weight: 700; }}
    .point {{ fill: #2f80ed; stroke: #173b57; stroke-width: 1.2; }}
    .accent {{ fill: #d64545; stroke: #6b1f1f; }}
    .trend {{ stroke: #c2410c; stroke-width: 3; stroke-dasharray: 8 6; }}
  </style>
  <text x="90" y="30" class="title">{html.escape(experiment_name)}: {label_type.title()} Training Distribution vs. Error Rate</text>
  <text x="90" y="52" class="subtitle">Training counts are log-scaled; error rates come from held-out constrained-speech evaluation.</text>
  <line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" class="axis"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" class="axis"/>
  {"".join(y_tick_elements)}
  {"".join(x_tick_elements)}
  <text x="304" y="505" class="axis-label">Training count per {label_type} (log scale)</text>
  <text x="20" y="300" class="axis-label" transform="rotate(-90 20 300)">{html.escape(y_axis_label)}</text>
  <line x1="{x_coord(min_x):.1f}" y1="{y_coord(line_y1):.1f}" x2="{x_coord(max_x):.1f}" y2="{y_coord(line_y2):.1f}" class="trend"/>
  <text x="500" y="96" class="subtitle">Pearson(log count, error) = {log_correlation:.3f}</text>
  <text x="500" y="116" class="subtitle">Spearman(count, error) = {spearman:.3f}</text>
  {"".join(point_elements)}
</svg>
'''
    svg_path = output_dir / f"{label_type}_training_distribution_vs_error.svg"
    svg_path.write_text(svg, encoding="utf-8")
    print(f"Saved {label_type} training distribution/error SVG to: {svg_path}")


def _ordered_labels(counter_map: dict[str, Counter[str]]) -> list[str]:
    """Collect and sort all labels appearing in a confusion map."""

    labels = set(counter_map.keys())
    for row in counter_map.values():
        labels.update(row.keys())
    labels.discard("")
    return sorted(labels)


def _ordered_vowel_labels(counter_map: dict[str, Counter[str]]) -> list[str]:
    """Return vowel labels in the declared inventory order plus any extras."""

    labels = set(_ordered_labels(counter_map))
    ordered = [label for label in VOWEL_LABEL_ORDER if label in labels]
    ordered.extend(sorted(label for label in labels if label not in ordered))
    return ordered


def _ordered_tone_labels(counter_map: dict[str, Counter[str]]) -> list[str]:
    """Return tone labels in the declared category order plus any extras."""

    labels = set(_ordered_labels(counter_map))
    ordered = [label for label in TONE_LABEL_ORDER if label in labels]
    ordered.extend(sorted(label for label in labels if label not in ordered))
    return ordered


def _analyze_rows(rows: Iterable[dict[str, str]]) -> dict[str, object]:
    """Compute vowel and tone confusion statistics over prediction rows.

    The analysis operates only on aligned vowel graphemes. Because tones are
    carried by vowels in this orthography, tone confusion is derived directly
    from aligned vowel units rather than from a separate tone string.
    """

    vowel_counts: dict[str, Counter[str]] = defaultdict(Counter)
    tone_counts: dict[str, Counter[str]] = defaultdict(Counter)
    tone_total = 0
    tone_substitutions = 0
    aligned_vowel_pairs = 0
    alignment_examples: list[dict[str, str]] = []

    for row in rows:
        reference = row.get("reference", "")
        prediction = row.get("prediction", "")
        ref_units = _extract_vowel_units(reference)
        hyp_units = _extract_vowel_units(prediction)
        alignment = _align_sequences(
            [unit["surface"] for unit in ref_units],
            [unit["surface"] for unit in hyp_units],
        )

        example_pairs = []
        ref_index = 0
        hyp_index = 0
        for ref_token, hyp_token in alignment:
            ref_unit = ref_units[ref_index] if ref_token is not None else None
            hyp_unit = hyp_units[hyp_index] if hyp_token is not None else None

            ref_vowel = ref_unit["vowel"] if ref_unit is not None else EPSILON
            hyp_vowel = hyp_unit["vowel"] if hyp_unit is not None else EPSILON
            vowel_counts[ref_vowel][hyp_vowel] += 1

            ref_tone = ref_unit["tone"] if ref_unit is not None else EPSILON
            hyp_tone = hyp_unit["tone"] if hyp_unit is not None else EPSILON
            tone_counts[ref_tone][hyp_tone] += 1

            if ref_unit is not None:
                tone_total += 1
                if ref_tone != hyp_tone:
                    tone_substitutions += 1

            aligned_vowel_pairs += 1
            if len(example_pairs) < 5:
                example_pairs.append(
                    {
                        "ref_token": ref_token or EPSILON,
                        "hyp_token": hyp_token or EPSILON,
                        "ref_vowel": ref_vowel,
                        "hyp_vowel": hyp_vowel,
                        "ref_tone": ref_tone,
                        "hyp_tone": hyp_tone,
                    }
                )

            if ref_token is not None:
                ref_index += 1
            if hyp_token is not None:
                hyp_index += 1

        if example_pairs and len(alignment_examples) < 10:
            alignment_examples.append(
                {
                    "utt_id": row.get("utt_id", ""),
                    "reference": reference,
                    "prediction": prediction,
                    "pairs": example_pairs,
                }
            )

    vowel_labels = _ordered_vowel_labels(vowel_counts)
    tone_labels = _ordered_tone_labels(tone_counts)
    return {
        "vowel_counts": vowel_counts,
        "tone_counts": tone_counts,
        "vowel_labels": vowel_labels,
        "tone_labels": tone_labels,
        "summary": {
            "aligned_vowel_pairs": aligned_vowel_pairs,
            "tone_total_reference_vowels": tone_total,
            "tone_substitutions": tone_substitutions,
            "tone_error_rate": float(tone_substitutions) / max(tone_total, 1),
        },
        "alignment_examples": alignment_examples,
    }


def _save_analysis(
    analysis: dict[str, object],
    predictions_csv: Path,
    output_dir: Path,
    training_manifest: Path | None = None,
    text_column: str | None = None,
) -> None:
    """Write CSV matrices, heatmaps, and summary outputs for one analysis result."""

    output_dir.mkdir(parents=True, exist_ok=True)

    _write_matrix(
        output_dir / "vowel_confusion_matrix.csv",
        analysis["vowel_labels"],
        analysis["vowel_labels"],
        analysis["vowel_counts"],
    )
    _write_matrix(
        output_dir / "tone_confusion_matrix.csv",
        analysis["tone_labels"],
        analysis["tone_labels"],
        analysis["tone_counts"],
    )
    vowel_distribution_rows = _distribution_error_rows(
        analysis["vowel_counts"],
        analysis["vowel_labels"],
        "vowel",
    )
    tone_distribution_rows = _distribution_error_rows(
        analysis["tone_counts"],
        analysis["tone_labels"],
        "tone",
    )
    _write_distribution_error_analysis(output_dir, vowel_distribution_rows, tone_distribution_rows)
    if training_manifest:
        training_rows = _read_csv_rows(training_manifest)
        selected_text_column = _select_text_column(training_rows, text_column)
        _write_training_distribution_error_analysis(
            output_dir,
            training_manifest,
            selected_text_column,
            vowel_distribution_rows,
            tone_distribution_rows,
        )
    summary_payload = {
        **analysis["summary"],
        "source_predictions_csv": str(predictions_csv),
        "vowel_labels": analysis["vowel_labels"],
        "tone_labels": analysis["tone_labels"],
        "alignment_examples": analysis["alignment_examples"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    vowel_heatmap_saved = _write_heatmap(
        output_dir / "vowel_confusion_heatmap.png",
        "Vowel Confusion Matrix",
        analysis["vowel_labels"],
        analysis["vowel_labels"],
        analysis["vowel_counts"],
    )
    tone_heatmap_saved = _write_heatmap(
        output_dir / "tone_confusion_heatmap.png",
        "Tone Confusion Matrix",
        analysis["tone_labels"],
        analysis["tone_labels"],
        analysis["tone_counts"],
    )

    print(f"Saved vowel confusion matrix to: {output_dir / 'vowel_confusion_matrix.csv'}")
    print(f"Saved tone confusion matrix to: {output_dir / 'tone_confusion_matrix.csv'}")
    if vowel_heatmap_saved:
        print(f"Saved vowel heatmap to: {output_dir / 'vowel_confusion_heatmap.png'}")
    else:
        print("Skipped vowel heatmap: matplotlib is not installed.")
    if tone_heatmap_saved:
        print(f"Saved tone heatmap to: {output_dir / 'tone_confusion_heatmap.png'}")
    else:
        print("Skipped tone heatmap: matplotlib is not installed.")
    print(f"Saved summary to: {output_dir / 'summary.json'}")
    print(f"Tone error rate: {summary_payload['tone_error_rate']:.4f}")


def _discover_predictions(results_root: Path) -> list[Path]:
    """Discover experiment prediction files under a results root."""

    return sorted(path for path in results_root.glob("*/predictions.csv") if path.is_file())


def main() -> None:
    """CLI entrypoint for single-file or batch vowel/tone confusion analysis."""

    parser = argparse.ArgumentParser(description="Analyze tone and vowel confusions from saved ASR predictions")
    parser.add_argument("predictions_csv", nargs="?", help="Path to a predictions.csv file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to a sibling folder named phonology_analysis",
    )
    parser.add_argument(
        "--all-experiments",
        action="store_true",
        help="Analyze all experiment predictions under --results-root at once",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Results root to scan when using --all-experiments",
    )
    parser.add_argument(
        "--training-manifest",
        default=None,
        help="Optional training manifest CSV used to compare training-unit frequency with held-out error rate",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Optional transcript column for --training-manifest. Defaults to transcription_norm, transcription, reference, or text.",
    )
    args = parser.parse_args()

    if args.all_experiments:
        results_root = Path(args.results_root).expanduser().resolve()
        predictions_files = _discover_predictions(results_root)
        if not predictions_files:
            raise FileNotFoundError(f"No predictions.csv files found under: {results_root}")
        for predictions_csv in predictions_files:
            print(f"Analyzing: {predictions_csv}")
            rows = _read_predictions(predictions_csv)
            analysis = _analyze_rows(rows)
            if args.output_dir:
                output_root = Path(args.output_dir).expanduser().resolve()
                output_dir = output_root / predictions_csv.parent.name
            else:
                output_dir = predictions_csv.parent / "phonology_analysis"
            training_manifest = Path(args.training_manifest).expanduser().resolve() if args.training_manifest else None
            _save_analysis(
                analysis,
                predictions_csv,
                output_dir,
                training_manifest=training_manifest,
                text_column=args.text_column,
            )
        return

    if not args.predictions_csv:
        raise ValueError("Provide predictions_csv for single-file mode, or use --all-experiments.")

    predictions_csv = Path(args.predictions_csv).expanduser().resolve()
    rows = _read_predictions(predictions_csv)
    analysis = _analyze_rows(rows)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = predictions_csv.parent / "phonology_analysis"
    training_manifest = Path(args.training_manifest).expanduser().resolve() if args.training_manifest else None
    _save_analysis(
        analysis,
        predictions_csv,
        output_dir,
        training_manifest=training_manifest,
        text_column=args.text_column,
    )


if __name__ == "__main__":
    main()
