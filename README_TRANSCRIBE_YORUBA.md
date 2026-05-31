# Yoruba Checkpoint Transcription

This is the minimal setup for transcribing Yoruba audio with Lateef's best Whisper checkpoint (This checkpoint achieves 0.3 CER):

`/workspace/storage/results/yoruba_pretrain/checkpoint-3520`

## 1) Install dependencies

```bash
pip install torch transformers librosa soundfile sentencepiece numpy
```

If audio decoding is incomplete on your machine, also install `ffmpeg`.

## 2) Transcribe one audio file

```bash
python scripts/transcribe_yoruba_checkpoint.py /path/to/audio.wav
```

## 3) Transcribe a folder of audio files

```bash
python scripts/transcribe_yoruba_checkpoint.py /path/to/audio_folder --output-csv yoruba_transcriptions.csv
```

## 4) Use a different checkpoint path

```bash
python scripts/transcribe_yoruba_checkpoint.py /path/to/audio.wav --checkpoint /path/to/checkpoint
```

## Notes

- The script forces Yoruba transcription with `language="yo"` and `task="transcribe"`.
- Supported audio extensions: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.opus`
- For a folder input, the script prints each transcription and can also save a CSV with:
  - `audio_path`
  - `transcription`
