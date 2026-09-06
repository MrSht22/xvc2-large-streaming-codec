# SA Generation on a New Server

This runbook reproduces the previous VoicePrivacy STTTS pair-generation
pipeline. Run one gate at a time. Do not launch the full dataset until the
single-file and 20-item pilots pass.

## Fixed Contract

```text
Conda environment: voiceprivacy-sa
Python:            3.10
PyTorch:           2.4.1 + CUDA 12.1
VoicePrivacy:      f59d6282fddd6c09d5dba5c261a180334ce2797d
STTTS models:      DigitalPhonetics speaker-anonymization v2.0
Backend:           sttts
SA level:          spk
Sample rate:       16 kHz mono
```

`spk` assigns one stable anonymous identity to each source speaker within a
run. Shards are speaker-safe, so one source speaker never crosses workers.

## 1. Create the Independent Environment

From the Codec repository root:

```bash
cd sa_generation
conda env create -f environment.yml
conda activate voiceprivacy-sa
python --version
```

Expected Python major/minor: `3.10`.

The environment still needs eSpeak-ng, CUDA PyTorch, and the pinned STTTS
Python dependencies. Follow `DEPLOYMENT.md` sections 2-4. The pip installer
uses the Tsinghua mirror by default.

## 2. Required External Assets

Do not commit these assets. Place them at:

```text
sa_generation/models/voicepat-v2/
sa_generation/nltk_data/
```

The model directory must contain `model_manifest.json` and the six files
checked by `verify_sa_setup.py`. NLTK must contain CMUdict plus both English
averaged-perceptron taggers. Set:

```bash
export NLTK_DATA="$PWD/nltk_data"
```

Persist that variable with a Conda activation hook after the files are in
place, as documented in `DEPLOYMENT.md`.

## 3. Environment Gate

```bash
python verify_sa_setup.py \
  --backend sttts \
  --models-dir models/voicepat-v2
```

Required final lines:

```text
model_integrity=PASS,files=6
setup_verification=PASS
```

## 4. Single-File Gate

```bash
mkdir -p runs

python anonymize_sa.py \
  --audio /absolute/path/to/test.wav \
  --utterance-id smoke001 \
  --speaker-id speaker001 \
  --backend sttts \
  --models-dir models/voicepat-v2 \
  --output-dir runs/sttts-smoke-one \
  --anonymization-level spk \
  --gpus 0 \
  --check-only
```

After `check_only=PASS`, run the same command against the same output directory
without `--check-only`. The input/settings fingerprint must remain unchanged.
Successful inference ends with `anonymization=PASS` and writes
`runs/sttts-smoke-one/pairs.jsonl`.

## 5. Twenty-Item Pilot

The generation manifest schema is:

```json
{"utterance_id":"...","speaker_id":"...","audio_path":"/absolute/path.wav","gender":"u"}
```

Use a deterministic 20-row subset of the final source manifest, then run:

```bash
python -u anonymize_sa.py \
  --manifest manifests/sa-pilot20.jsonl \
  --backend sttts \
  --models-dir models/voicepat-v2 \
  --output-dir runs/sa-pilot20-sttts \
  --anonymization-level spk \
  --gpus 0 \
  2>&1 | tee runs/sa-pilot20-sttts.log
```

Check that `pairs.jsonl` has 20 rows and manually listen to several source/SA
pairs before scaling up.

## 6. Parallel Generation

The previous completed topology was four GPUs with four workers per GPU. That
is fast but can run out of memory. Start the new hardware conservatively with
one worker per GPU, then increase only after measuring peak VRAM:

```bash
python -u parallel_anonymize_sa.py \
  --manifest manifests/sa-train.jsonl \
  --models-dir models/voicepat-v2 \
  --output-dir runs/sa-train-sttts-spk \
  --gpus 0,1,2,3 \
  --workers-per-gpu 1 \
  --anonymization-level spk \
  --seed 2026 \
  2>&1 | tee runs/sa-train-sttts-spk-launcher.log
```

The launcher writes `parallel_run_metadata.json`, speaker-safe shard
manifests, per-shard caches/logs, and the final ordered `pairs.jsonl`.

## 7. Resume an Interrupted Run

Do not start a second launcher while old workers are still alive. After all
old workers exit, inspect first:

```bash
python -u resume_parallel_anonymize_sa.py \
  --output-dir runs/sa-train-sttts-spk \
  --check-only
```

Then resume incomplete shards:

```bash
python -u resume_parallel_anonymize_sa.py \
  --output-dir runs/sa-train-sttts-spk \
  --gpus 0,1,2,3 \
  2>&1 | tee -a runs/sa-train-sttts-spk-resume.log
```

The recovery path verifies the original manifest hash, keeps the original
shard assignment and seed, skips complete shards, reuses intermediate caches,
and merges output in original manifest order.

## Output Used by Codec Preprocessing

Each generated row contains:

```text
utterance_id
speaker_id
original_audio_path
normalized_source_audio_path
anonymized_audio_path
backend
anonymization_level
source_audio
anonymized_audio
```

Codec pair preprocessing should use `normalized_source_audio_path` and
`anonymized_audio_path`. STTTS is a resynthesis system, so source and SA audio
are not guaranteed to be sample-aligned and must pass the later pair-quality
filter before strong frame-level supervision.
