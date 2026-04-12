import os
from pathlib import Path
import pandas as pd

# =========================
# CONFIG
# =========================
INPUT_FILE = "metadata_with_all_splits_clean.csv"
OUTPUT_ROOT = Path("data") / "metadata"

# Column names in your cleaned master file
MAIN_COL = "split"
REVERSE_COL = "split_reverse_aux"
MIXED_CONSTRAINED_COL = "split_mixed_to_constrained_aux"

SPLIT_VALUES = ["train", "dev", "test"]


def validate_columns(df: pd.DataFrame) -> None:
    required_cols = [
        "utt_id",
        "speaker_id",
        "session_id",
        "style",
        "style_bin",
        MAIN_COL,
        REVERSE_COL,
        MIXED_CONSTRAINED_COL,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def export_split_set(
    df: pd.DataFrame,
    split_col: str,
    out_dir: Path,
    include_style_subsets: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Exporting {split_col} -> {out_dir} ===")

    for split in SPLIT_VALUES:
        subset = df[df[split_col] == split].copy()
        out_path = out_dir / f"{split}.csv"
        subset.to_csv(out_path, index=False)
        print(f"Saved {out_path} ({len(subset)} rows)")

        if include_style_subsets:
            for style_val in sorted(subset["style_bin"].dropna().unique()):
                style_subset = subset[subset["style_bin"] == style_val].copy()
                safe_style = str(style_val).replace(" ", "_")
                style_path = out_dir / f"{split}_{safe_style}.csv"
                style_subset.to_csv(style_path, index=False)
                print(f"  Saved {style_path} ({len(style_subset)} rows)")


def print_diagnostics(df: pd.DataFrame, split_col: str) -> None:
    print(f"\n=== Diagnostics for {split_col} ===")
    print("\nCounts by split:")
    print(df[split_col].value_counts(dropna=False).sort_index())

    print("\nStyle x split:")
    print(pd.crosstab(df[split_col], df["style_bin"], dropna=False))

    # Speaker leakage check
    split_speakers = {
        split: set(df.loc[df[split_col] == split, "speaker_id"].dropna().unique())
        for split in SPLIT_VALUES
    }

    print("\nSpeaker overlap:")
    print("train ∩ dev :", split_speakers["train"] & split_speakers["dev"])
    print("train ∩ test:", split_speakers["train"] & split_speakers["test"])
    print("dev ∩ test  :", split_speakers["dev"] & split_speakers["test"])

    # Utterance overlap check
    split_utts = {
        split: set(df.loc[df[split_col] == split, "utt_id"].dropna().unique())
        for split in SPLIT_VALUES
    }

    print("\nUtterance overlap:")
    print("train ∩ dev :", split_utts["train"] & split_utts["dev"])
    print("train ∩ test:", split_utts["train"] & split_utts["test"])
    print("dev ∩ test  :", split_utts["dev"] & split_utts["test"])


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Could not find input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    validate_columns(df)

    print(f"Loaded {INPUT_FILE} with {len(df)} rows.")

    # Export MAIN split
    print_diagnostics(df, MAIN_COL)
    export_split_set(
        df=df,
        split_col=MAIN_COL,
        out_dir=OUTPUT_ROOT / "main",
        include_style_subsets=True,
    )

    # Export REVERSE AUX split
    print_diagnostics(df, REVERSE_COL)
    export_split_set(
        df=df,
        split_col=REVERSE_COL,
        out_dir=OUTPUT_ROOT / "reverse_aux",
        include_style_subsets=True,
    )

    # Export MIXED -> CONSTRAINED AUX split
    print_diagnostics(df, MIXED_CONSTRAINED_COL)
    export_split_set(
        df=df,
        split_col=MIXED_CONSTRAINED_COL,
        out_dir=OUTPUT_ROOT / "mixed_to_constrained_aux",
        include_style_subsets=True,
    )

    print("\nDone.")
    print(f"All files saved under: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()