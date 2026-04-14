from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List


class DataValidationError(ValueError):
    pass


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate_required_columns(rows: List[Dict[str, str]], required_columns: Iterable[str], manifest_path: Path) -> None:
    if not rows:
        raise DataValidationError(f"Manifest is empty: {manifest_path}")

    present = set(rows[0].keys())
    missing = [col for col in required_columns if col not in present]
    if missing:
        raise DataValidationError(f"Missing required columns in {manifest_path}: {missing}")


def resolve_audio_path(audio_path_value: str, project_root: Path) -> Path:
    raw = Path(audio_path_value).expanduser()
    if raw.is_absolute():
        return raw
    return (project_root / raw).resolve()


def validate_audio_paths_exist(rows: List[Dict[str, str]], audio_path_column: str, project_root: Path) -> None:
    missing = []
    for row in rows:
        resolved = resolve_audio_path(row[audio_path_column], project_root=project_root)
        if not resolved.exists():
            missing.append(row[audio_path_column])
    if missing:
        preview = missing[:10]
        raise FileNotFoundError(
            f"Referenced audio files are missing ({len(missing)} total). "
            f"Examples: {preview}"
        )
