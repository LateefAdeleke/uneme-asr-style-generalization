Master metadata:
data/metadata/master/metadata_with_all_splits_clean.csv

Split folders:
data/metadata/main/
data/metadata/reverse_aux/
data/metadata/mixed_to_constrained_aux/

Audio roots:
data/audio/
data/careful_audio/

All scripts should read audio paths from the `audio_path` column and should not hardcode alternate audio directories.

Important:
- Always read paths from the audio_path column
- Do not hardcode alternate audio directories
- Main split is the primary benchmark
- reverse_aux and mixed_to_constrained_aux are supplementary