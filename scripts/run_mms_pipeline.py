#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr_pipeline.runner import run_pipeline


ALLOWED_EXPERIMENTS = {"E1", "E2", "E3", "E4", "E5"}
EXPECTED_SPLIT_REGIME_BY_EXPERIMENT = {
    "E1": "main",
    "E2": "main",
    "E3": "main",
    "E4": "reverse_aux",
    "E5": "mixed_to_constrained_aux",
}


def _parse_experiments(raw: str) -> list[str]:
    exps = [item.strip() for item in raw.split(",") if item.strip()]
    if not exps:
        raise ValueError("At least one experiment must be provided.")
    invalid = [exp for exp in exps if exp not in ALLOWED_EXPERIMENTS]
    if invalid:
        raise ValueError(f"Only E1,E2,E3,E4,E5 are supported for MMS validation. Invalid: {invalid}")
    return exps


def build_runtime_config(args: argparse.Namespace) -> dict:
    experiments = _parse_experiments(args.experiments)
    runtime = {
        "project_root": str(args.project_root),
        "registry_path": args.registry,
        "experiments": experiments,
        "expected_split_regime": "main",
        "expected_split_regime_by_experiment": {
            exp: EXPECTED_SPLIT_REGIME_BY_EXPERIMENT[exp] for exp in experiments
        },
        "required_columns": [
            "audio_path",
            "utt_id",
            "speaker_id",
            "session_id",
            "style",
            "style_bin",
        ],
        "text_column_override": args.text_column,
        "fallback_text_columns": ["transcription_norm", "transcription"],
        "utt_id_column": "utt_id",
        "speaker_id_column": "speaker_id",
        "session_id_column": "session_id",
        "style_column": "style",
        "validate_audio_paths": not args.skip_audio_check,
        "smoke_test": args.smoke_test,
        "smoke_max_rows": args.smoke_max_rows,
        "logs_root": args.logs_root,
        "aggregate_metrics_path": args.aggregate_metrics,
        "model_family_override": "mms_ctc",
        "experiment_id_suffix": "_mms_ctc",
        "output_dir_suffix": "_mms_ctc",
        "model_name_or_path": args.model_name_or_path,
        "processor_name_or_path": args.processor_name_or_path,
        "target_lang": args.target_lang,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "evaluation_strategy": args.evaluation_strategy,
        "save_strategy": args.save_strategy,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
    }
    if args.use_fp16 is not None:
        runtime["use_fp16"] = args.use_fp16
    return runtime


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Run MMS CTC pipeline for E1/E2/E3/E4/E5")
    parser.add_argument("--project-root", type=Path, default=repo_root)
    parser.add_argument("--registry", default="configs/experiment_registry.yaml")
    parser.add_argument("--experiments", default="E1,E2,E3,E4,E5")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--logs-root", default="results/logs")
    parser.add_argument("--aggregate-metrics", default="results/aggregate_metrics_mms_e1_e5.json")

    parser.add_argument("--model-name-or-path", default="facebook/mms-1b-all")
    parser.add_argument("--processor-name-or-path", default="facebook/mms-1b-all")
    parser.add_argument("--target-lang", default=None, help="Optional MMS adapter language code for supported languages")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--evaluation-strategy", default="epoch")
    parser.add_argument("--save-strategy", default="epoch")
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    fp16_group = parser.add_mutually_exclusive_group()
    fp16_group.add_argument("--use-fp16", dest="use_fp16", action="store_true")
    fp16_group.add_argument("--no-fp16", dest="use_fp16", action="store_false")
    parser.set_defaults(use_fp16=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-rows", type=int, default=8)
    parser.add_argument("--skip-audio-check", action="store_true")
    args = parser.parse_args()

    runtime = build_runtime_config(args)
    tmp_runtime = Path(args.project_root) / "results" / "runtime_config_mms_e1_e5.json"
    tmp_runtime.parent.mkdir(parents=True, exist_ok=True)
    tmp_runtime.write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    run_pipeline(str(tmp_runtime))


if __name__ == "__main__":
    main()
