# Upstream source lock

The vendored source in `third_party/Voice-Privacy-Challenge-2026` is based on:

- Repository: https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2026
- Commit: `f59d6282fddd6c09d5dba5c261a180334ce2797d`
- Retrieved: 2026-08-08
- License: GNU GPL v3 or later (see the vendored `LICENSE` file)

One local runtime patch is intentionally retained in
`anonymization/modules/sttts/tts/IMSToucan/Preprocessing/AudioPreprocessor.py`:
very short inputs are padded only for pyloudnorm analysis, while the waveform
passed to synthesis keeps its original length. This prevents the previously
observed `Audio must have length greater than the block size` shard failure.

The neural STTTS checkpoints are the public v2.0 release from:

- https://github.com/DigitalPhonetics/speaker-anonymization/releases/tag/v2.0

This deployment wrapper is separate from the vendored source. It prepares
arbitrary WAV/JSONL input, fixes deterministic identity mapping, keeps model
files read-only, and writes an explicit source/anonymized pair manifest.
