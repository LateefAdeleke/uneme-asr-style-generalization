from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config_loader import load_runtime_config, load_simple_yaml
from .data import (
    load_manifest,
    resolve_audio_path,
    validate_audio_paths_exist,
    validate_required_columns,
)
from .metrics import cer, wer
from .modeling import (
    CTCTrainingConfig,
    MinimalCTCBackend,
    MinimalWhisperBackend,
    RealCTCBackend,
    RealWhisperBackend,
    WhisperTrainingConfig,
)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_predictions_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_repo_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _select_text_column(data_cfg: Dict[str, Any], runtime: Dict[str, Any], sample_row: Dict[str, str]) -> str:
    candidates = []
    if runtime.get("text_column_override"):
        candidates.append(runtime["text_column_override"])
    candidates.append(data_cfg["text_column"])
    candidates.extend(runtime.get("fallback_text_columns", []))

    seen = set()
    for col in candidates:
        if col in seen:
            continue
        seen.add(col)
        if col in sample_row:
            return col
    raise KeyError(f"None of the candidate text columns were found in manifest columns: {candidates}")


def _resolve_transfer_checkpoint(
    project_root: Path,
    transfer_cfg: Dict[str, Any],
    transfer_condition: str,
    *,
    allow_placeholder: bool,
) -> str | None:
    if transfer_condition != "cross_lingual_transfer":
        return None

    checkpoint = transfer_cfg.get("source_checkpoint")
    if not checkpoint or checkpoint == "REPLACE_WITH_SOURCE_CHECKPOINT":
        if allow_placeholder:
            return None
        raise ValueError(
            "cross_lingual_transfer requires a concrete source_checkpoint in configs/experiment_registry.yaml"
        )

    checkpoint_path = Path(str(checkpoint))
    if checkpoint_path.is_absolute():
        return str(checkpoint_path)

    repo_relative_path = (project_root / checkpoint_path).resolve()
    if repo_relative_path.exists():
        return str(repo_relative_path)

    return str(checkpoint)


def _build_whisper_training_config(
    registry: Dict[str, Any],
    runtime: Dict[str, Any],
    *,
    smoke_test: bool,
    project_root: Path,
    exp: Dict[str, Any],
) -> WhisperTrainingConfig:
    training_cfg = registry["training"]
    model_cfg = registry["models"]["whisper"]
    transfer_condition = exp["transfer_condition"]
    transfer_cfg = registry["transfer_setups"][transfer_condition]
    init_model_name_or_path = _resolve_transfer_checkpoint(
        project_root,
        transfer_cfg,
        transfer_condition,
        allow_placeholder=smoke_test,
    )
    base_model_name_or_path = runtime.get("model_name_or_path", model_cfg["pretrained_model_name_or_path"])
    return WhisperTrainingConfig(
        model_name_or_path=base_model_name_or_path,
        init_model_name_or_path=init_model_name_or_path or base_model_name_or_path,
        freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
        language=runtime.get("language", model_cfg.get("language")),
        task=runtime.get("task", model_cfg.get("task", "transcribe")),
        transfer_condition=transfer_condition,
        learning_rate=float(runtime.get("learning_rate", training_cfg["learning_rate"])),
        weight_decay=float(runtime.get("weight_decay", training_cfg["weight_decay"])),
        warmup_ratio=float(runtime.get("warmup_ratio", training_cfg["warmup_ratio"])),
        per_device_train_batch_size=int(
            runtime.get("per_device_train_batch_size", training_cfg["per_device_train_batch_size"])
        ),
        per_device_eval_batch_size=int(runtime.get("per_device_eval_batch_size", training_cfg["per_device_eval_batch_size"])),
        gradient_accumulation_steps=int(
            runtime.get("gradient_accumulation_steps", training_cfg["gradient_accumulation_steps"])
        ),
        num_train_epochs=float(runtime.get("num_train_epochs", training_cfg["num_train_epochs"])),
        eval_strategy=str(runtime.get("evaluation_strategy", training_cfg["evaluation_strategy"])),
        save_strategy=str(runtime.get("save_strategy", training_cfg["save_strategy"])),
        logging_strategy=str(runtime.get("logging_strategy", training_cfg["logging_strategy"])),
        logging_steps=int(runtime.get("logging_steps", training_cfg["logging_steps"])),
        save_total_limit=int(runtime.get("save_total_limit", training_cfg["save_total_limit"])),
        fp16=bool(runtime.get("use_fp16", model_cfg.get("use_fp16", True))),
        seed=int(runtime.get("seed", registry["defaults"]["random_seed"])),
    )


def _build_ctc_training_config(
    registry: Dict[str, Any],
    runtime: Dict[str, Any],
) -> CTCTrainingConfig:
    training_cfg = registry["training"]
    model_cfg = registry["models"]["wav2vec2_ctc"]
    model_name_or_path = runtime.get("model_name_or_path", model_cfg["pretrained_model_name_or_path"])
    processor_name_or_path = runtime.get("processor_name_or_path", model_cfg["processor_name_or_path"])
    return CTCTrainingConfig(
        model_name_or_path=model_name_or_path,
        processor_name_or_path=processor_name_or_path,
        init_model_name_or_path=model_name_or_path,
        learning_rate=float(runtime.get("learning_rate", training_cfg["learning_rate"])),
        weight_decay=float(runtime.get("weight_decay", training_cfg["weight_decay"])),
        warmup_ratio=float(runtime.get("warmup_ratio", training_cfg["warmup_ratio"])),
        per_device_train_batch_size=int(
            runtime.get("per_device_train_batch_size", training_cfg["per_device_train_batch_size"])
        ),
        per_device_eval_batch_size=int(runtime.get("per_device_eval_batch_size", training_cfg["per_device_eval_batch_size"])),
        gradient_accumulation_steps=int(
            runtime.get("gradient_accumulation_steps", training_cfg["gradient_accumulation_steps"])
        ),
        num_train_epochs=float(runtime.get("num_train_epochs", training_cfg["num_train_epochs"])),
        eval_strategy=str(runtime.get("evaluation_strategy", training_cfg["evaluation_strategy"])),
        save_strategy=str(runtime.get("save_strategy", training_cfg["save_strategy"])),
        logging_strategy=str(runtime.get("logging_strategy", training_cfg["logging_strategy"])),
        logging_steps=int(runtime.get("logging_steps", training_cfg["logging_steps"])),
        save_total_limit=int(runtime.get("save_total_limit", training_cfg["save_total_limit"])),
        fp16=bool(runtime.get("use_fp16", model_cfg.get("use_fp16", True))),
        seed=int(runtime.get("seed", registry["defaults"]["random_seed"])),
    )


def _apply_suffix(value: str, suffix: str | None) -> str:
    if not suffix:
        return value
    path = Path(value)
    if path.parent == Path("."):
        return f"{path.name}{suffix}"
    return str(path.parent / f"{path.name}{suffix}")


def run_pipeline(runtime_config_path: str) -> List[Dict[str, Any]]:
    runtime = load_runtime_config(runtime_config_path)
    project_root = Path(runtime.get("project_root", Path.cwd())).resolve()
    registry_path = _resolve_repo_path(project_root, runtime["registry_path"])
    registry = load_simple_yaml(registry_path)

    data_cfg = registry["data"]
    experiments_cfg = registry["experiments"]

    selected = runtime["experiments"]
    smoke_test = runtime.get("smoke_test", False)
    smoke_max_rows = int(runtime.get("smoke_max_rows", 8))
    validate_audio = bool(runtime.get("validate_audio_paths", True))
    expected_split_regime = runtime.get("expected_split_regime", "main")
    expected_split_regime_by_experiment = runtime.get("expected_split_regime_by_experiment", {})

    base_required_columns = runtime["required_columns"]
    audio_col = data_cfg["audio_path_column"]
    style_bin_col = data_cfg.get("style_bin_column", "style_bin")
    model_family_override = runtime.get("model_family_override")
    experiment_id_suffix = runtime.get("experiment_id_suffix")
    output_dir_suffix = runtime.get("output_dir_suffix")

    results: List[Dict[str, Any]] = []
    for exp_key in selected:
        exp = experiments_cfg[exp_key]

        if exp_key not in {"E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E1f", "E2f", "E3f"}:
            raise ValueError(f"Unsupported experiment for this pipeline: {exp_key}")

        model_family = str(model_family_override or exp["model_family"])
        if model_family not in {"whisper", "wav2vec2_ctc"}:
            raise ValueError(f"Unsupported model_family for this pipeline: {model_family}")

        expected_regime = expected_split_regime_by_experiment.get(exp_key, expected_split_regime)
        if exp["split_regime"] != expected_regime:
            raise ValueError(
                f"Unexpected split regime for {exp_key}: {exp['split_regime']} (expected {expected_regime})"
            )
        if exp["transfer_condition"] not in registry["transfer_setups"]:
            raise ValueError(f"Unknown transfer_condition for {exp_key}: {exp['transfer_condition']}")

        train_manifest = _resolve_repo_path(project_root, exp["train_manifest"])
        dev_manifest = _resolve_repo_path(project_root, exp["dev_manifest"])
        test_manifest = _resolve_repo_path(project_root, exp["test_manifest"])

        train_rows = load_manifest(train_manifest)
        dev_rows = load_manifest(dev_manifest)
        test_rows = load_manifest(test_manifest)

        text_col = _select_text_column(data_cfg, runtime, train_rows[0])
        required_columns = list(dict.fromkeys([*base_required_columns, text_col]))

        validate_required_columns(train_rows, required_columns, train_manifest)
        validate_required_columns(dev_rows, required_columns, dev_manifest)
        validate_required_columns(test_rows, required_columns, test_manifest)

        if smoke_test:
            train_rows = train_rows[:smoke_max_rows]
            dev_rows = dev_rows[:smoke_max_rows]
            test_rows = test_rows[:smoke_max_rows]

        if validate_audio:
            validate_audio_paths_exist(train_rows, audio_col, project_root=project_root)
            validate_audio_paths_exist(dev_rows, audio_col, project_root=project_root)
            validate_audio_paths_exist(test_rows, audio_col, project_root=project_root)

        train_audio_paths = [str(resolve_audio_path(r[audio_col], project_root=project_root)) for r in train_rows]
        dev_audio_paths = [str(resolve_audio_path(r[audio_col], project_root=project_root)) for r in dev_rows]
        test_audio_paths = [str(resolve_audio_path(r[audio_col], project_root=project_root)) for r in test_rows]

        experiment_id = _apply_suffix(exp["experiment_id"], experiment_id_suffix)
        output_dir = _resolve_repo_path(project_root, _apply_suffix(exp["output_dir"], output_dir_suffix))
        logs_dir = _resolve_repo_path(project_root, runtime["logs_root"]) / experiment_id
        logs_dir.mkdir(parents=True, exist_ok=True)

        if smoke_test:
            backend = MinimalWhisperBackend() if model_family == "whisper" else MinimalCTCBackend()
            backend.train(train_rows, dev_rows)
            predictions = backend.predict(test_rows)
            test_metrics = {
                "wer": wer([r[text_col] for r in test_rows], predictions),
                "cer": cer([r[text_col] for r in test_rows], predictions),
            }
            model_name_or_path = runtime.get("model_name_or_path", registry["models"][model_family]["pretrained_model_name_or_path"])
            model_initialization = model_name_or_path
        else:
            if model_family == "whisper":
                whisper_cfg = _build_whisper_training_config(
                    registry,
                    runtime,
                    smoke_test=smoke_test,
                    project_root=project_root,
                    exp=exp,
                )
                backend = RealWhisperBackend(whisper_cfg)
                predictions, test_metrics = backend.train_and_predict(
                    train_rows=train_rows,
                    dev_rows=dev_rows,
                    test_rows=test_rows,
                    train_audio_paths=train_audio_paths,
                    dev_audio_paths=dev_audio_paths,
                    test_audio_paths=test_audio_paths,
                    text_column=text_col,
                    output_dir=output_dir,
                )
                model_name_or_path = whisper_cfg.model_name_or_path
                model_initialization = whisper_cfg.init_model_name_or_path
            else:
                ctc_cfg = _build_ctc_training_config(registry, runtime)
                backend = RealCTCBackend(ctc_cfg)
                predictions, test_metrics = backend.train_and_predict(
                    train_rows=train_rows,
                    dev_rows=dev_rows,
                    test_rows=test_rows,
                    train_audio_paths=train_audio_paths,
                    dev_audio_paths=dev_audio_paths,
                    test_audio_paths=test_audio_paths,
                    text_column=text_col,
                    output_dir=output_dir,
                )
                model_name_or_path = ctc_cfg.model_name_or_path
                model_initialization = ctc_cfg.init_model_name_or_path

        pred_rows = []
        for row, pred in zip(test_rows, predictions):
            pred_rows.append(
                {
                    "experiment_id": experiment_id,
                    "utt_id": row.get(runtime["utt_id_column"], ""),
                    "speaker_id": row.get(runtime["speaker_id_column"], ""),
                    "session_id": row.get(runtime["session_id_column"], ""),
                    "style_bin": row.get(style_bin_col, ""),
                    "reference": row.get(text_col, ""),
                    "prediction": pred,
                }
            )

        metrics = {
            "experiment_id": experiment_id,
            "model_family": model_family,
            "split_regime": exp["split_regime"],
            "transfer_condition": exp["transfer_condition"],
            "model_initialization": model_initialization,
            "text_column_used": text_col,
            "n_test": len(test_rows),
            "wer": float(test_metrics["wer"]),
            "cer": float(test_metrics["cer"]),
            "smoke_test": smoke_test,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        _write_json(output_dir / "metrics.json", metrics)
        _write_predictions_csv(output_dir / "predictions.csv", pred_rows)
        _write_predictions_jsonl(output_dir / "predictions.jsonl", pred_rows)
        (logs_dir / "run.log").write_text(
            (
                f"Completed {experiment_id} with n_test={len(test_rows)} smoke={smoke_test} "
                f"text_column={text_col} model_family={model_family} model={model_name_or_path} "
                f"transfer={exp['transfer_condition']} init={model_initialization}\n"
            ),
            encoding="utf-8",
        )

        results.append(metrics)

    aggregate_path = _resolve_repo_path(project_root, runtime["aggregate_metrics_path"])
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
