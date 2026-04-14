#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from asr_pipeline.yoruba_adaptation import run_yoruba_adaptation


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Run Yoruba Whisper adaptation for transfer initialization")
    parser.add_argument("--project-root", type=Path, default=repo_root)
    parser.add_argument("--config", default="configs/yoruba_adaptation.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-rows", type=int, default=8)
    args = parser.parse_args()

    run_yoruba_adaptation(
        project_root=args.project_root,
        config_path=args.config,
        smoke_test=args.smoke_test,
        smoke_max_rows=args.smoke_max_rows,
    )


if __name__ == "__main__":
    main()
