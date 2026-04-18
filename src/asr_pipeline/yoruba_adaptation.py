from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .config_loader import load_simple_yaml
from .metrics import cer, wer
from .modeling import MinimalWhisperBackend, RealWhisperBackend, WhisperTrainingConfig


def _log(message: str) -> None:
    print(message, flush=True)


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


def _format_registry_checkpoint(project_root: Path, checkpoint: str) -> str:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        return checkpoint
    try:
        relative_path = checkpoint_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return checkpoint
    return relative_path.as_posix()


def _validate_dataset_config(dataset_cfg: Dict[str, Any]) -> None:
    if dataset_cfg.get("source_type") != "huggingface":
        raise ValueError(f"Unsupported Yoruba dataset source_type: {dataset_cfg.get('source_type')}")
    has_dataset_name = dataset_cfg.get("dataset_name") not in {None, "", "YOUR_IROYINSPEECH_DATASET_ID"}
    has_dataset_script = dataset_cfg.get("dataset_script_path") not in {None, ""}
    if not has_dataset_name and not has_dataset_script:
        raise ValueError(
            "configs/yoruba_adaptation.yaml must set either dataset.dataset_name to the public IroyinSpeech dataset ID "
            "or dataset.dataset_script_path to a dataset loading script"
        )


def _load_public_dataset(project_root: Path, dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    from datasets import load_dataset

    _validate_dataset_config(dataset_cfg)
    dataset_path = dataset_cfg.get("dataset_script_path")
    kwargs: Dict[str, Any] = {
        "path": str(_resolve_repo_path(project_root, dataset_path)) if dataset_path else dataset_cfg["dataset_name"],
        "trust_remote_code": bool(dataset_cfg.get("trust_remote_code", False)),
    }
    if dataset_cfg.get("dataset_config_name"):
        kwargs["name"] = dataset_cfg["dataset_config_name"]
    if dataset_cfg.get("data_dir"):
        kwargs["data_dir"] = dataset_cfg["data_dir"]
    return load_dataset(**kwargs)


def _resolve_split(dataset_dict: Dict[str, Any], split_name: str | None) -> Any | None:
    if not split_name:
        return None
    if split_name not in dataset_dict:
        raise KeyError(f"Requested split '{split_name}' not found. Available splits: {list(dataset_dict.keys())}")
    return dataset_dict[split_name]


def _limit_split(split: Any | None, smoke_test: bool, smoke_max_rows: int) -> Any | None:
    if split is None:
        return None
    if not smoke_test:
        return split
    return split.select(range(min(len(split), smoke_max_rows)))


def _require_columns(split: Any, *, split_name: str, required_columns: Iterable[str]) -> None:
    present = set(split.column_names)
    missing = [column for column in required_columns if column not in present]
    if missing:
        raise KeyError(f"Missing required columns in Yoruba adaptation split '{split_name}': {missing}")


def _inspect_dataset(
    dataset_dict: Dict[str, Any],
    *,
    audio_column: str,
    transcript_column: str,
    speaker_id_column: str | None,
    train_split_name: str,
    dev_split_name: str | None,
    test_split_name: str | None,
) -> Dict[str, Any]:
    from datasets import Audio

    required_columns = [audio_column, transcript_column]
    if speaker_id_column:
        required_columns.append(speaker_id_column)

    inspection: Dict[str, Any] = {
        "split_sizes": {},
        "columns_by_split": {},
        "decoded_audio_examples": [],
    }

    split_specs = [
        ("train", train_split_name),
        ("validation", dev_split_name),
        ("test", test_split_name),
    ]
    first_available_split = None
    for label, split_name in split_specs:
        if not split_name:
            continue
        split = _resolve_split(dataset_dict, split_name)
        _require_columns(split, split_name=split_name, required_columns=required_columns)
        inspection["split_sizes"][label] = len(split)
        inspection["columns_by_split"][label] = list(split.column_names)
        _log(f"Yoruba adaptation {label} split '{split_name}' size: {len(split)}")
        _log(f"Yoruba adaptation {label} columns: {', '.join(split.column_names)}")
        if first_available_split is None and len(split) > 0:
            first_available_split = (split_name, split.cast_column(audio_column, Audio(sampling_rate=16000)))

    if first_available_split is None:
        raise ValueError("No non-empty split available for Yoruba adaptation audio inspection")

    source_split_name, decoded_split = first_available_split
    num_examples = min(2, len(decoded_split))
    for idx in range(num_examples):
        example = decoded_split[idx]
        audio = example[audio_column]
        audio_array = audio.get("array")
        sampling_rate = audio.get("sampling_rate")
        if audio_array is None or sampling_rate is None:
            raise ValueError(
                f"Failed to decode Yoruba adaptation audio example {idx} from split '{source_split_name}'"
            )
        num_samples = len(audio_array)
        transcript_preview = str(example[transcript_column])[:80]
        inspection["decoded_audio_examples"].append(
            {
                "split": source_split_name,
                "index": idx,
                "sampling_rate": int(sampling_rate),
                "num_samples": int(num_samples),
                "transcript_preview": transcript_preview,
            }
        )
        _log(
            f"Decoded audio example {idx + 1}/{num_examples} from '{source_split_name}': "
            f"sampling_rate={sampling_rate}, num_samples={num_samples}, transcript='{transcript_preview}'"
        )

    return inspection


def _column_or_default(example: Dict[str, Any], column_name: str | None, default_value: str) -> str:
    if column_name and column_name in example and example[column_name] is not None:
        return str(example[column_name])
    return default_value


def _build_dataset_from_hf_split(
    backend: RealWhisperBackend,
    split: Any,
    *,
    audio_column: str,
    transcript_column: str,
):
    from datasets import Audio
    import numpy as np

    if audio_column not in split.column_names:
        raise KeyError(f"Missing audio column '{audio_column}' in dataset columns: {split.column_names}")
    if transcript_column not in split.column_names:
        raise KeyError(f"Missing transcript column '{transcript_column}' in dataset columns: {split.column_names}")

    split = split.cast_column(audio_column, Audio(sampling_rate=16000))
    processor = backend.processor

    def _prep(example: Dict[str, Any]) -> Dict[str, List[int] | List[float]]:
        audio = example[audio_column]
        audio_array = np.asarray(audio["array"], dtype=np.float32)
        input_features = processor.feature_extractor(
            audio_array,
            sampling_rate=int(audio["sampling_rate"]),
        ).input_features[0]
        labels = processor.tokenizer(str(example[transcript_column])).input_ids
        return {
            "input_features": input_features,
            "labels": labels,
        }

    return split.map(_prep, remove_columns=split.column_names)


def _decode_predictions(
    backend: RealWhisperBackend,
    pred_output: Any,
) -> List[str]:
    pred_ids = pred_output.predictions[0] if isinstance(pred_output.predictions, tuple) else pred_output.predictions
    if getattr(pred_ids, "ndim", 0) == 3:
        pred_ids = pred_ids.argmax(axis=-1)
    return backend.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)


def _build_prediction_rows(
    split: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    *,
    transcript_column: str,
    utt_id_column: str | None,
    speaker_id_column: str | None,
    split_name: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for idx, (example, prediction) in enumerate(zip(split, predictions)):
        rows.append(
            {
                "split": split_name,
                "utt_id": _column_or_default(example, utt_id_column, f"{split_name}_{idx}"),
                "speaker_id": _column_or_default(example, speaker_id_column, ""),
                "reference": str(example[transcript_column]),
                "prediction": prediction,
            }
        )
    return rows


def _predict_and_score(
    trainer: Any,
    backend: RealWhisperBackend,
    split: Any | None,
    *,
    audio_column: str,
    split_name: str,
    transcript_column: str,
    utt_id_column: str | None,
    speaker_id_column: str | None,
) -> tuple[Dict[str, Any], List[Dict[str, str]]]:
    if split is None or len(split) == 0:
        return {}, []

    processed_split = _build_dataset_from_hf_split(
        backend,
        split,
        audio_column=audio_column,
        transcript_column=transcript_column,
    )
    pred_output = trainer.predict(processed_split)
    predictions = _decode_predictions(backend, pred_output)
    refs = [str(example[transcript_column]) for example in split]
    metrics = {
        f"{split_name}_wer": float(wer(refs, predictions)),
        f"{split_name}_cer": float(cer(refs, predictions)),
        f"{split_name}_loss": float(pred_output.metrics.get(f"test_loss", 0.0)),
        f"n_{split_name}": len(split),
    }
    pred_rows = _build_prediction_rows(
        split,
        predictions,
        transcript_column=transcript_column,
        utt_id_column=utt_id_column,
        speaker_id_column=speaker_id_column,
        split_name=split_name,
    )
    return metrics, pred_rows


def _build_training_config(config: Dict[str, Any]) -> WhisperTrainingConfig:
    model_cfg = config["model"]
    training_cfg = config["training"]
    return WhisperTrainingConfig(
        model_name_or_path=model_cfg["base_model_name_or_path"],
        init_model_name_or_path=model_cfg["base_model_name_or_path"],
        freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
        language=model_cfg.get("language"),
        task=model_cfg.get("task", "transcribe"),
        transfer_condition="yoruba_adaptation",
        learning_rate=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        warmup_ratio=float(training_cfg["warmup_ratio"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        eval_strategy=str(training_cfg["evaluation_strategy"]),
        save_strategy=str(training_cfg["save_strategy"]),
        logging_strategy=str(training_cfg["logging_strategy"]),
        logging_steps=int(training_cfg["logging_steps"]),
        save_total_limit=int(training_cfg["save_total_limit"]),
        fp16=bool(model_cfg.get("use_fp16", True)),
        seed=int(training_cfg["seed"]),
    )


def run_yoruba_adaptation(
    *,
    project_root: Path,
    config_path: str | Path,
    smoke_test: bool = False,
    smoke_max_rows: int = 8,
) -> Dict[str, Any]:
    config = load_simple_yaml(_resolve_repo_path(project_root, str(config_path)))
    run_cfg = config["run"]
    dataset_cfg = config["dataset"]
    runtime_cfg = config.get("runtime", {})
    effective_smoke_test = bool(smoke_test or runtime_cfg.get("smoke_test", False))
    effective_smoke_max_rows = int(runtime_cfg.get("smoke_max_rows", smoke_max_rows))

    output_dir = _resolve_repo_path(project_root, run_cfg["output_dir"])
    logs_dir = _resolve_repo_path(project_root, run_cfg["logs_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = _load_public_dataset(project_root, dataset_cfg)
    inspection = _inspect_dataset(
        dataset_dict,
        audio_column=dataset_cfg["audio_column"],
        transcript_column=dataset_cfg["transcript_column"],
        speaker_id_column=dataset_cfg.get("speaker_id_column"),
        train_split_name=dataset_cfg.get("train_split"),
        dev_split_name=dataset_cfg.get("dev_split"),
        test_split_name=dataset_cfg.get("test_split"),
    )
    train_split = _limit_split(_resolve_split(dataset_dict, dataset_cfg.get("train_split")), effective_smoke_test, effective_smoke_max_rows)
    dev_split = _limit_split(_resolve_split(dataset_dict, dataset_cfg.get("dev_split")), effective_smoke_test, effective_smoke_max_rows)
    test_split = _limit_split(_resolve_split(dataset_dict, dataset_cfg.get("test_split")), effective_smoke_test, effective_smoke_max_rows)

    transcript_column = dataset_cfg["transcript_column"]
    audio_column = dataset_cfg["audio_column"]
    utt_id_column = dataset_cfg.get("utt_id_column")
    speaker_id_column = dataset_cfg.get("speaker_id_column")

    if effective_smoke_test:
        backend = MinimalWhisperBackend()
        backend.train(list(train_split), list(dev_split) if dev_split is not None else [])
        dev_predictions = backend.predict(list(dev_split)) if dev_split is not None else []
        test_predictions = backend.predict(list(test_split)) if test_split is not None else []

        metrics = {
            "experiment_id": run_cfg["experiment_id"],
            "dataset_name": dataset_cfg["dataset_name"],
            "base_model_name_or_path": config["model"]["base_model_name_or_path"],
            "best_checkpoint": config["model"]["base_model_name_or_path"],
            "smoke_test": True,
            "dataset_inspection": inspection,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if dev_split is not None:
            dev_refs = [str(example[transcript_column]) for example in dev_split]
            metrics["dev_wer"] = float(wer(dev_refs, dev_predictions))
            metrics["dev_cer"] = float(cer(dev_refs, dev_predictions))
            metrics["n_dev"] = len(dev_split)
        if test_split is not None:
            test_refs = [str(example[transcript_column]) for example in test_split]
            metrics["test_wer"] = float(wer(test_refs, test_predictions))
            metrics["test_cer"] = float(cer(test_refs, test_predictions))
            metrics["n_test"] = len(test_split)

        dev_rows = _build_prediction_rows(
            list(dev_split) if dev_split is not None else [],
            dev_predictions,
            transcript_column=transcript_column,
            utt_id_column=utt_id_column,
            speaker_id_column=speaker_id_column,
            split_name="dev",
        )
        test_rows = _build_prediction_rows(
            list(test_split) if test_split is not None else [],
            test_predictions,
            transcript_column=transcript_column,
            utt_id_column=utt_id_column,
            speaker_id_column=speaker_id_column,
            split_name="test",
        )
    else:
        backend = RealWhisperBackend(_build_training_config(config))
        train_ds = _build_dataset_from_hf_split(
            backend,
            train_split,
            audio_column=audio_column,
            transcript_column=transcript_column,
        )
        dev_ds = (
            _build_dataset_from_hf_split(
                backend,
                dev_split,
                audio_column=audio_column,
                transcript_column=transcript_column,
            )
            if dev_split is not None and len(dev_split) > 0
            else None
        )

        from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments

        training_cfg = config["training"]
        has_eval = dev_ds is not None and len(dev_ds) > 0
        eval_strategy = str(training_cfg["evaluation_strategy"]) if has_eval else "no"
        save_strategy = str(training_cfg["save_strategy"]) if has_eval else "no"

        def compute_metrics(pred: Any) -> Dict[str, float]:
            predictions = _decode_predictions(backend, pred)
            label_ids = pred.label_ids
            import numpy as np

            label_ids = np.where(label_ids == -100, backend.processor.tokenizer.pad_token_id, label_ids)
            refs = backend.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            return {"wer": wer(refs, predictions), "cer": cer(refs, predictions)}

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
            learning_rate=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg["weight_decay"]),
            warmup_ratio=float(training_cfg["warmup_ratio"]),
            num_train_epochs=float(training_cfg["num_train_epochs"]),
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            logging_strategy=str(training_cfg["logging_strategy"]),
            logging_steps=int(training_cfg["logging_steps"]),
            save_total_limit=int(training_cfg["save_total_limit"]),
            fp16=bool(config["model"].get("use_fp16", True)),
            predict_with_generate=True,
            generation_max_length=225,
            remove_unused_columns=False,
            label_names=["labels"],
            report_to=[],
            seed=int(training_cfg["seed"]),
            load_best_model_at_end=has_eval,
            metric_for_best_model=str(training_cfg.get("metric_for_best_model", "wer")),
            greater_is_better=bool(training_cfg.get("greater_is_better", False)),
        )

        callbacks: List[Any] = []
        if has_eval and "early_stopping_patience" in training_cfg:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(training_cfg["early_stopping_patience"])))

        trainer = Seq2SeqTrainer(
            model=backend.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds if has_eval else None,
            data_collator=backend._data_collator(),
            processing_class=backend.processor,
            compute_metrics=compute_metrics if has_eval else None,
            callbacks=callbacks,
        )
        trainer.train()

        best_checkpoint = trainer.state.best_model_checkpoint or str(output_dir)
        best_checkpoint = _format_registry_checkpoint(project_root, best_checkpoint)
        metrics = {
            "experiment_id": run_cfg["experiment_id"],
            "dataset_name": dataset_cfg["dataset_name"],
            "base_model_name_or_path": config["model"]["base_model_name_or_path"],
            "best_checkpoint": best_checkpoint,
            "smoke_test": False,
            "dataset_inspection": inspection,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        dev_metrics, dev_rows = _predict_and_score(
            trainer,
            backend,
            dev_split,
            audio_column=audio_column,
            split_name="dev",
            transcript_column=transcript_column,
            utt_id_column=utt_id_column,
            speaker_id_column=speaker_id_column,
        )
        test_metrics, test_rows = _predict_and_score(
            trainer,
            backend,
            test_split,
            audio_column=audio_column,
            split_name="test",
            transcript_column=transcript_column,
            utt_id_column=utt_id_column,
            speaker_id_column=speaker_id_column,
        )
        metrics.update(dev_metrics)
        metrics.update(test_metrics)

    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "best_checkpoint.json", {"best_checkpoint": metrics["best_checkpoint"]})
    if dev_rows:
        _write_predictions_csv(output_dir / "predictions_dev.csv", dev_rows)
        _write_predictions_jsonl(output_dir / "predictions_dev.jsonl", dev_rows)
    if test_rows:
        _write_predictions_csv(output_dir / "predictions_test.csv", test_rows)
        _write_predictions_jsonl(output_dir / "predictions_test.jsonl", test_rows)

    assumptions = {
        "dataset_access": (
            "The public Yoruba source corpus is accessible through datasets.load_dataset using either "
            "dataset.dataset_name or dataset.dataset_script_path if a dataset-specific loader is required."
        ),
        "transcript_column": transcript_column,
        "audio_column": audio_column,
        "audio_handling": "The dataset exposes a Hugging Face audio column that can be cast to 16kHz during preprocessing.",
        "split_availability": {
            "train_split": dataset_cfg.get("train_split"),
            "dev_split": dataset_cfg.get("dev_split"),
            "test_split": dataset_cfg.get("test_split"),
        },
    }
    _write_json(output_dir / "assumptions.json", assumptions)
    (logs_dir / "run.log").write_text(
        (
            f"Completed {run_cfg['experiment_id']} smoke={effective_smoke_test} "
            f"dataset={dataset_cfg['dataset_name']} base_model={config['model']['base_model_name_or_path']} "
            f"best_checkpoint={metrics['best_checkpoint']}\n"
        ),
        encoding="utf-8",
    )
    return metrics
