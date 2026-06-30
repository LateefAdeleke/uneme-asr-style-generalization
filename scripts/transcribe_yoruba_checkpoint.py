#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


DEFAULT_CHECKPOINT = "/workspace/storage/results/yoruba_pretrain/checkpoint-3520"
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}


def _load_model(checkpoint: str, device: str | None) -> tuple[WhisperProcessor, WhisperForConditionalGeneration, str]:
    processor = WhisperProcessor.from_pretrained(checkpoint)
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="yo", task="transcribe")
    model.generation_config.forced_decoder_ids = forced_decoder_ids

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()
    return processor, model, device


def _transcribe_file(
    audio_path: Path,
    *,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: str,
) -> str:
    audio_array, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    audio_array = np.asarray(audio_array, dtype=np.float32)
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def _collect_audio_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not files:
            raise FileNotFoundError(f"No supported audio files found in directory: {path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _write_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "transcription"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe Yoruba audio with a fine-tuned Whisper checkpoint")
    parser.add_argument("input_path", help="Path to one audio file or a folder of audio files")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Whisper checkpoint path")
    parser.add_argument("--output-csv", default=None, help="Optional CSV output path for batch transcription")
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu or cuda")
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    files = _collect_audio_files(input_path)
    processor, model, device = _load_model(args.checkpoint, args.device)

    rows: list[dict[str, str]] = []
    for audio_file in files:
        transcription = _transcribe_file(audio_file, processor=processor, model=model, device=device)
        rows.append({"audio_path": str(audio_file), "transcription": transcription})
        print(f"{audio_file}\n{transcription}\n")

    if args.output_csv:
        _write_csv(Path(args.output_csv).expanduser().resolve(), rows)


if __name__ == "__main__":
    main()
