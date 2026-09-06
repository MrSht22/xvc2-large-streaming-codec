# VoicePrivacy SA pair generator

This directory generates normalized source/anonymized WAV pairs without
integrating them into X-VC. It provides two citable VoicePrivacy methods:

- `mcadams`: VoicePrivacy 2022 baseline B2. It is CPU-only and has no model
  weights.
- `sttts`: the phonetic STTTS/GAN system submitted as T04 to VoicePrivacy
  2022, using the maintained VoicePrivacy 2026 implementation and the public
  DigitalPhonetics v2.0 checkpoints. The same method family is called B3 in
  VoicePrivacy 2026; it was not a 2022 baseline named B3.

See [RESEARCH_REPORT.md](RESEARCH_REPORT.md) for provenance and citations and
[DEPLOYMENT.md](DEPLOYMENT.md) for the complete Linux server procedure.

## Entry points

- `anonymize_sa.py`: single-WAV or JSONL batch anonymization.
- `parallel_anonymize_sa.py`: speaker-safe multi-process STTTS inference across GPUs.
- `prepare_audio_manifest.py`: recursively index an audio directory for batch inference.
- `prepare_models.py`: CRC-check, safely extract, and hash STTTS archives.
- `verify_sa_setup.py`: validate dependencies, vendored source, and models.

The vendored upstream snapshot is locked in [UPSTREAM.md](UPSTREAM.md).

## Input

For one file:

```bash
python anonymize_sa.py \
  --audio /path/input.wav \
  --utterance-id utt-001 \
  --speaker-id speaker-001 \
  --backend mcadams \
  --output-dir runs/mcadams-smoke
```

For a batch, provide JSONL with one object per line:

```json
{"utterance_id":"utt-001","speaker_id":"speaker-001","audio_path":"/data/a.wav","gender":"u"}
```

```bash
python prepare_audio_manifest.py \
  --audio-root /data/LibriTTS/train-clean-100 \
  --speaker-parent-depth 2 \
  --output manifests/train-clean-100.jsonl

python anonymize_sa.py \
  --manifest manifests/train-clean-100.jsonl \
  --backend sttts \
  --models-dir models/voicepat-v2 \
  --output-dir runs/sttts \
  --gpus 0
```

For four-GPU STTTS generation, use one independent process per GPU. The
launcher keeps every speaker in one shard, isolates intermediate directories,
and merges the successful shard outputs in source-manifest order:

```bash
python -u parallel_anonymize_sa.py \
  --manifest manifests/train-clean-100-all.jsonl \
  --models-dir models/voicepat-v2 \
  --output-dir runs/train-clean-100-sttts-4gpu \
  --gpus 0,1,2,3 \
  --workers-per-gpu 1 \
  --anonymization-level spk
```

Worker logs are written under `<output-dir>/logs/`. Rerunning the exact command
resumes the shard output directories. The merged index is
`<output-dir>/pairs.jsonl`.

`--anonymization-level spk` keeps one pseudonymous mapping per speaker within
the run. Use `utt` to choose a new target per utterance.

`--speaker-parent-depth 1` means the immediate parent directory identifies the
speaker. LibriTTS uses `speaker/chapter/audio.wav`, so its speaker depth is 2.

## Output

Both backends write `pairs.jsonl`. Every row contains the original path, the
normalized 16 kHz mono source, the anonymized WAV, IDs, backend settings, and
audio metadata. STTTS is a resynthesis system, so output duration is not
sample-aligned with the source. McAdams preserves the normalized waveform
length.

An output directory is bound to an input/settings fingerprint. Use a new
directory if the input or configuration changes.
