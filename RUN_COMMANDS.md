# Run Commands for Whisper E1 / E2 / E3 / E4 / E5

This file provides exact local commands for running the current Whisper pipeline.

The primary benchmark is the `main` split (`E1-E3`). Supplementary diagnostics use the auxiliary split regimes:

- `E4` uses `reverse_aux`
- `E5` uses `mixed_to_constrained_aux`

Run all commands from the project root.

## 1) Environment setup

### Linux/macOS
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch transformers datasets numpy
```

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch transformers datasets numpy
```

If your audio files are not all plain WAV/PCM, install `ffmpeg` as well.

## 2) Lowest-cost first check

### bash
```bash
PYTHONPATH=src python scripts/run_whisper_pipeline.py --experiments E1 --smoke-test --skip-audio-check --smoke-max-rows 2
```

### PowerShell
```powershell
$env:PYTHONPATH = "src"
python scripts/run_whisper_pipeline.py --experiments E1 --smoke-test --skip-audio-check --smoke-max-rows 2
```

This validates config loading, split selection, manifest parsing, output writing, and command wiring with tiny cost.

## 3) Smoke-test commands

### E1 smoke
```powershell
$env:PYTHONPATH = "src"
python scripts/run_whisper_pipeline.py --experiments E1 --smoke-test --skip-audio-check --smoke-max-rows 8
```

### E2 smoke
```powershell
$env:PYTHONPATH = "src"
python scripts/run_whisper_pipeline.py --experiments E2 --smoke-test --skip-audio-check --smoke-max-rows 8
```

### E3 smoke
```powershell
$env:PYTHONPATH = "src"
python scripts/run_whisper_pipeline.py --experiments E3 --smoke-test --skip-audio-check --smoke-max-rows 8
```

### E4 smoke
```powershell
$env:PYTHONPATH = "src"
python scripts/run_whisper_pipeline.py --experiments E4 --smoke-test --skip-audio-check --smoke-max-rows 8
```

### E5 smoke
```powershell
$env:PYTHONPATH = "src"
python scripts/run_whisper_pipeline.py --experiments E5 --smoke-test --skip-audio-check --smoke-max-rows 8
```

## 4) Real training commands

These commands assume your machine has audio files at paths referenced by `audio_path` in metadata.

### E1 real training
```powershell
$env:PYTHONPATH = "src"
python scripts/run_whisper_pipeline.py `
  --experiments E1 `
  --project-root . `
  --registry configs/experiment_registry.yaml `
  --model-name-or-path openai/whisper-small `
  --num-train-epochs 3 `
  --per-device-train-batch-size 8 `
  --per-device-eval-batch-size 8 `
  --learning-rate 1e-4
```

### E2 real training
```powershell
$env:PYTHONPATH = "src"
python scripts/run_whisper_pipeline.py `
  --experiments E2 `
  --project-root . `
  --registry configs/experiment_registry.yaml `
  --model-name-or-path openai/whisper-small `
  --num-train-epochs 3 `
  --per-device-train-batch-size 8 `
  --per-device-eval-batch-size 8 `
  --learning-rate 1e-4
```

### E3 real training
```powershell
$env:PYTHONPATH = "src"
python scripts/run_whisper_pipeline.py `
  --experiments E3 `
  --project-root . `
  --registry configs/experiment_registry.yaml `
  --model-name-or-path openai/whisper-small `
  --num-train-epochs 3 `
  --per-device-train-batch-size 8 `
  --per-device-eval-batch-size 8 `
  --learning-rate 1e-4
```

### E4 real training
```powershell
$env:PYTHONPATH = "src"
python scripts/run_whisper_pipeline.py `
  --experiments E4 `
  --project-root . `
  --registry configs/experiment_registry.yaml `
  --model-name-or-path openai/whisper-small `
  --num-train-epochs 3 `
  --per-device-train-batch-size 8 `
  --per-device-eval-batch-size 8 `
  --learning-rate 1e-4
```

### E5 real training
```powershell
$env:PYTHONPATH = "src"
python scripts/run_whisper_pipeline.py `
  --experiments E5 `
  --project-root . `
  --registry configs/experiment_registry.yaml `
  --model-name-or-path openai/whisper-small `
  --num-train-epochs 3 `
  --per-device-train-batch-size 8 `
  --per-device-eval-batch-size 8 `
  --learning-rate 1e-4
```

For single-line use in PowerShell, this also works:

```powershell
$env:PYTHONPATH = "src"; python scripts/run_whisper_pipeline.py --experiments E4 --project-root . --registry configs/experiment_registry.yaml --model-name-or-path openai/whisper-small --num-train-epochs 3 --per-device-train-batch-size 8 --per-device-eval-batch-size 8 --learning-rate 1e-4
```

## 5) Windows notes

1. Open PowerShell inside the repository root.
2. Set `PYTHONPATH` with `$env:PYTHONPATH = "src"`.
3. For multi-line PowerShell commands, use the backtick character `` ` `` for continuation, not `\`.
4. Forward slashes in Python CLI arguments are fine on Windows.
5. If policy blocks activation scripts, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 6) Expected outputs

For each experiment, outputs are written to the experiment `output_dir` from the registry:

- `E1`: `results/E1_nat2nat_main_noXfer/`
- `E2`: `results/E2_cons2nat_main_noXfer/`
- `E3`: `results/E3_mix2nat_main_noXfer/`
- `E4`: `results/E4_nat2cons_rev_noXfer/`
- `E5`: `results/E5_mix2cons_aux_noXfer/`

Inside each experiment directory:

- `metrics.json`
- `predictions.csv`
- `predictions.jsonl`

Logs:

- `results/logs/<experiment_id>/run.log`

Aggregate metrics file default:

- `results/aggregate_metrics_e1_e3.json`

If you want a filename that reflects all five experiments:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_whisper_pipeline.py --experiments E1,E2,E3,E4,E5 --aggregate-metrics results/aggregate_metrics_e1_e5.json
```

## 7) FAQ

### Required dependencies

- `torch`
- `transformers`
- `datasets`
- `numpy`

### Is `ffmpeg` required?

- Recommended for broad audio format support during decoding.
- If all audio is standard WAV readable by your installed backend, it may not be needed, but having it installed helps avoid decode failures.

### Required environment variables

- No mandatory environment variables beyond `PYTHONPATH=src` unless you install the package another way.

### Can I run multiple experiments together?

Yes. The current pipeline accepts any comma-separated subset of `E1,E2,E3,E4,E5`.

```powershell
$env:PYTHONPATH = "src"
python scripts/run_whisper_pipeline.py --experiments E1,E2,E3,E4,E5 --smoke-test --skip-audio-check --smoke-max-rows 8
```
