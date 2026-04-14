# Run Commands for Whisper E1 / E2 / E3

This file provides **exact local commands** for running the current Whisper pipeline.

> Run all commands from the project root:
> `.../uneme-asr-style-generalization`

---

## 1) Environment setup (required)

### 1.1 Create and activate a virtual environment

**Linux/macOS (bash/zsh):**
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 1.2 Install Python dependencies required by the current pipeline

```bash
pip install torch transformers datasets numpy
```

If your audio files are not all plain WAV/PCM, install ffmpeg on your OS as well (see notes below).

---

## 2) Lowest-cost first check (run this first)

Use this command first to verify the pipeline wiring end-to-end at minimal cost:

```bash
PYTHONPATH=src python scripts/run_whisper_pipeline.py --experiments E1 --smoke-test --skip-audio-check --smoke-max-rows 2
```

This validates config loading, split selection, manifest parsing, output writing, and command wiring with tiny cost.

---

## 3) Smoke-test commands (one per experiment)

### E1 smoke
```bash
PYTHONPATH=src python scripts/run_whisper_pipeline.py --experiments E1 --smoke-test --skip-audio-check --smoke-max-rows 8
```

### E2 smoke
```bash
PYTHONPATH=src python scripts/run_whisper_pipeline.py --experiments E2 --smoke-test --skip-audio-check --smoke-max-rows 8
```

### E3 smoke
```bash
PYTHONPATH=src python scripts/run_whisper_pipeline.py --experiments E3 --smoke-test --skip-audio-check --smoke-max-rows 8
```

---

## 4) Real training commands (one per experiment)

> These commands assume your local machine has audio files at paths referenced by `audio_path` in metadata.

### E1 real training
```bash
PYTHONPATH=src python scripts/run_whisper_pipeline.py \
  --experiments E1 \
  --project-root . \
  --registry configs/experiment_registry.yaml \
  --model-name-or-path openai/whisper-small \
  --num-train-epochs 3 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 8 \
  --learning-rate 1e-4
```

### E2 real training
```bash
PYTHONPATH=src python scripts/run_whisper_pipeline.py \
  --experiments E2 \
  --project-root . \
  --registry configs/experiment_registry.yaml \
  --model-name-or-path openai/whisper-small \
  --num-train-epochs 3 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 8 \
  --learning-rate 1e-4
```

### E3 real training
```bash
PYTHONPATH=src python scripts/run_whisper_pipeline.py \
  --experiments E3 \
  --project-root . \
  --registry configs/experiment_registry.yaml \
  --model-name-or-path openai/whisper-small \
  --num-train-epochs 3 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 8 \
  --learning-rate 1e-4
```

---

## 5) Windows project-root notes

1. Open PowerShell **inside the repository root**.
2. Use `set` command equivalent for PYTHONPATH in PowerShell:
   ```powershell
   $env:PYTHONPATH = "src"
   python scripts/run_whisper_pipeline.py --experiments E1 --smoke-test --skip-audio-check --smoke-max-rows 2
   ```
3. Keep path separators in CLI arguments as shown (forward slashes work with Python on Windows in these commands).
4. If policy blocks activation scripts, run:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

---

## 6) Expected outputs and save locations

For each experiment, outputs are written to the experiment's `output_dir` from the registry:

- E1: `results/E1_nat2nat_main_noXfer/`
- E2: `results/E2_cons2nat_main_noXfer/`
- E3: `results/E3_mix2nat_main_noXfer/`

Inside each experiment directory:
- `metrics.json`
- `predictions.csv`
- `predictions.jsonl`

Logs:
- `results/logs/<experiment_id>/run.log`

Aggregate metrics file (default):
- `results/aggregate_metrics_e1_e3.json`

---

## 7) Dependency and runtime FAQ

### Exact dependencies required by current Whisper pipeline
- `torch`
- `transformers`
- `datasets`
- `numpy`

### Is ffmpeg required?
- **Recommended / often required** for broad audio format support during dataset decoding.
- If all audio is standard WAV readable by your installed backend, ffmpeg may not be used, but keeping ffmpeg installed avoids decode failures.

### Any required environment variables?
- No mandatory environment variables are required by this pipeline.
- `PYTHONPATH=src` (or `$env:PYTHONPATH="src"` on PowerShell) is required unless you install the package another way.

### What command should I run first?
Run this first:
```bash
PYTHONPATH=src python scripts/run_whisper_pipeline.py --experiments E1 --smoke-test --skip-audio-check --smoke-max-rows 2
```
