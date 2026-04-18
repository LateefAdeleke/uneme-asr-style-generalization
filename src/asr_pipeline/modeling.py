from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import soundfile as sf

from .metrics import cer, wer


class MinimalWhisperBackend:
    """Deterministic placeholder backend for smoke/repro checks only."""

    def train(self, train_rows: List[Dict[str, str]], dev_rows: List[Dict[str, str]]) -> None:
        _ = (train_rows, dev_rows)

    def predict(self, test_rows: List[Dict[str, str]]) -> List[str]:
        return ["" for _ in test_rows]


@dataclass
class WhisperTrainingConfig:
    model_name_or_path: str
    init_model_name_or_path: str
    freeze_encoder: bool
    language: str | None
    task: str
    transfer_condition: str
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    eval_strategy: str
    save_strategy: str
    logging_strategy: str
    logging_steps: int
    save_total_limit: int
    fp16: bool
    seed: int


class RealWhisperBackend:
    """Whisper fine-tuning backend using Hugging Face Seq2SeqTrainer."""

    def __init__(self, cfg: WhisperTrainingConfig):
        self.cfg = cfg

        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as exc:  # pragma: no cover - dependency-driven
            raise ImportError(
                "Real Whisper training requires `torch`, `transformers`, and `datasets` to be installed."
            ) from exc

        self.torch = torch
        self.WhisperProcessor = WhisperProcessor
        self.WhisperForConditionalGeneration = WhisperForConditionalGeneration

        self.processor = WhisperProcessor.from_pretrained(cfg.model_name_or_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(cfg.init_model_name_or_path)

        if cfg.freeze_encoder:
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False

            n_trainable = sum(p.requires_grad for p in self.model.parameters())
            n_total = sum(1 for _ in self.model.parameters())
            print(f"[DEBUG] Trainable params: {n_trainable}/{n_total}")

            encoder_trainable = any(p.requires_grad for p in self.model.model.encoder.parameters())
            print(f"[DEBUG] Encoder trainable: {encoder_trainable}")

        if cfg.language:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=cfg.language,
                task=cfg.task,
            )
            self.model.generation_config.forced_decoder_ids = forced_decoder_ids

        self.model.config.use_cache = False

    def _build_dataset(
        self,
        rows: Sequence[Dict[str, str]],
        audio_paths: Sequence[str],
        text_column: str,
    ):
        from datasets import Dataset
        import librosa
        import numpy as np

        examples = []
        for row, resolved_audio_path in zip(rows, audio_paths):
            item = dict(row)
            item["audio_path"] = resolved_audio_path
            examples.append(item)

        ds = Dataset.from_list(examples)
        processor = self.processor
        target_sr = 16000

        def _prep(example: Dict[str, str]) -> Dict[str, List[int] | List[float]]:
            audio_array, sampling_rate = sf.read(example["audio_path"])

            if hasattr(audio_array, "ndim") and audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)

            audio_array = np.asarray(audio_array, dtype=np.float32)

            if sampling_rate != target_sr:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sampling_rate,
                    target_sr=target_sr,
                )
                sampling_rate = target_sr

            input_features = processor.feature_extractor(
                audio_array,
                sampling_rate=sampling_rate,
            ).input_features[0]

            labels = processor.tokenizer(example[text_column]).input_ids

            return {
                "input_features": input_features,
                "labels": labels,
            }

        return ds.map(_prep, remove_columns=ds.column_names)

    def _data_collator(self):
        processor = self.processor
        decoder_start_token_id = self.model.config.decoder_start_token_id

        class DataCollatorSpeechSeq2SeqWithPadding:
            def __call__(self, features):
                input_features = [{"input_features": feature["input_features"]} for feature in features]
                batch = processor.feature_extractor.pad(input_features, return_tensors="pt")

                label_features = [{"input_ids": feature["labels"]} for feature in features]
                labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")
                labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

                if (labels[:, 0] == decoder_start_token_id).all().cpu().item():
                    labels = labels[:, 1:]

                batch["labels"] = labels
                return batch

        return DataCollatorSpeechSeq2SeqWithPadding()

    def train_and_predict(
        self,
        train_rows: Sequence[Dict[str, str]],
        dev_rows: Sequence[Dict[str, str]],
        test_rows: Sequence[Dict[str, str]],
        train_audio_paths: Sequence[str],
        dev_audio_paths: Sequence[str],
        test_audio_paths: Sequence[str],
        text_column: str,
        output_dir: Path,
    ) -> tuple[List[str], Dict[str, float]]:
        import numpy as np
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

        train_ds = self._build_dataset(train_rows, train_audio_paths, text_column=text_column)
        dev_ds = (
            self._build_dataset(dev_rows, dev_audio_paths, text_column=text_column)
            if len(dev_rows) > 0 and len(dev_audio_paths) > 0
            else None
        )
        test_ds = self._build_dataset(test_rows, test_audio_paths, text_column=text_column)

        processor = self.processor

        def compute_metrics(pred):
            pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
            if pred_ids.ndim == 3:
                pred_ids = np.argmax(pred_ids, axis=-1)

            label_ids = pred.label_ids
            label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

            pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            return {"wer": wer(label_str, pred_str), "cer": cer(label_str, pred_str)}

        has_eval = dev_ds is not None and len(dev_ds) > 0
        eval_strategy = self.cfg.eval_strategy if has_eval else "no"
        save_strategy = self.cfg.save_strategy if has_eval else "no"

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            learning_rate=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            warmup_ratio=self.cfg.warmup_ratio,
            num_train_epochs=self.cfg.num_train_epochs,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            logging_strategy=self.cfg.logging_strategy,
            logging_steps=self.cfg.logging_steps,
            save_total_limit=self.cfg.save_total_limit,
            fp16=self.cfg.fp16,
            predict_with_generate=True,
            generation_max_length=225,
            remove_unused_columns=False,
            label_names=["labels"],
            report_to=[],
            seed=self.cfg.seed,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds if has_eval else None,
            data_collator=self._data_collator(),
            processing_class=processor,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        pred_output = trainer.predict(test_ds)
        pred_ids = pred_output.predictions[0] if isinstance(pred_output.predictions, tuple) else pred_output.predictions
        if pred_ids.ndim == 3:
            pred_ids = pred_ids.argmax(axis=-1)

        predictions = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        refs = [row[text_column] for row in test_rows]
        metrics = {
            "wer": wer(refs, predictions),
            "cer": cer(refs, predictions),
            "test_loss": float(pred_output.metrics.get("test_loss", 0.0)),
        }
        return predictions, metrics
