from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_simple_yaml(path: str | Path) -> Dict[str, Any]:
    """Load simple nested YAML mappings (sufficient for this repo registry)."""
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]

    with Path(path).open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            indent = len(line) - len(line.lstrip(" "))
            key_value = line.strip()
            if ":" not in key_value:
                continue

            key, rest = key_value.split(":", 1)
            key = key.strip()
            rest = rest.strip()

            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]

            if rest == "":
                parent[key] = {}
                stack.append((indent, parent[key]))
            else:
                parent[key] = _parse_scalar(rest)

    return root


def load_runtime_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
