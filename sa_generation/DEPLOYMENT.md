# Linux server deployment

The commands below assume this directory is uploaded as
`voiceprivacy_sa_deployment/`. No Git clone is required at deployment time
because the tested VoicePrivacy source snapshot is vendored.

## 1. Upload

Required for both backends:

```text
voiceprivacy_sa_deployment/
```

Required only for STTTS: upload either the three archives below or the already
prepared extracted directory and its `model_manifest.json`.

```text
anonymization.zip
asr.zip
tts.zip
```

Their roles are:

| Archive | Contents | Used for |
|---|---|---|
| `anonymization.zip` | style-embedding WGAN | Generate artificial target-speaker embeddings |
| `asr.zip` | Branchformer hybrid CTC/attention phone ASR | Convert source speech into a phonetic sequence |
| `tts.zip` | aligner, GST embedding model, FastSpeech2 and HiFi-GAN | Extract phone-level prosody and synthesize anonymized speech |

Do not upload the local `tmp/sa-sttts-env` environment. It contains macOS ARM
binaries and cannot run on a Linux x86_64 server.

## 2. Create the environment

The Conda file uses the Anaconda `defaults` channel only. Its Linux packages
provide FFmpeg, libsndfile, PortAudio, and all build tools without sudo.
`sox` is not used by this verified inference path. The remaining eSpeak-ng
dependency is installed into the Conda prefix in the next step.

Create the Python environment:

```bash
cd /path/to/uploaded/voiceprivacy_sa_deployment
conda env create -f environment.yml
conda activate voiceprivacy-sa
```

Install eSpeak-ng without sudo. Upload the local artifact directory
`espeak-ng-1.52.0-source/`, then run:

```bash
sha256sum -c /path/to/espeak-ng-1.52.0-source/SHA256SUMS

bash install_espeak_ng_user.sh \
  /path/to/espeak-ng-1.52.0-source/espeak-ng-1.52.0.tar.gz

conda deactivate
conda activate voiceprivacy-sa
```

The installer builds only the text/phoneme functionality needed by
`phonemizer`, installs the executable, shared library, and language data under
`$CONDA_PREFIX`, and creates a Conda activation hook for
`PHONEMIZER_ESPEAK_LIBRARY`.

Install PyTorch before the remaining STTTS packages. For a CUDA 12.1 server:

```bash
python -m pip install \
  torch==2.4.1+cu121 \
  torchaudio==2.4.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

bash install_sttts_requirements.sh
python -m pip check
```

If the server cannot reliably reach the PyTorch index, upload the complete
pre-downloaded wheel directory
`pytorch-2.4.1-cu121-linux-x86_64-py310/` and install offline:

```bash
export PYTORCH_WHEEL_DIR=/path/to/pytorch-2.4.1-cu121-linux-x86_64-py310

cd "$PYTORCH_WHEEL_DIR"
sha256sum -c SHA256SUMS

cd /path/to/voiceprivacy_sa_deployment
python -m pip install \
  --no-index \
  --find-links "$PYTORCH_WHEEL_DIR" \
  "torch==2.4.1+cu121" \
  "torchaudio==2.4.1+cu121"

bash install_sttts_requirements.sh
python -m pip check
```

Upload all wheels in that directory, not only the two files named `torch` and
`torchaudio`: PyTorch's CUDA runtime, cuDNN, cuBLAS, NCCL, Triton, and Python
dependencies are separate wheels.

For CPU-only STTTS inference:

```bash
python -m pip install torch==2.4.1 torchaudio==2.4.1
bash install_sttts_requirements.sh
```

For McAdams only, the smaller environment is sufficient:

```bash
python -m pip install -r requirements-mcadams.txt
```

The neural dependency pins are intentional: ESPnet 202310 requires NumPy
below 1.24, and the tested combination is NumPy 1.23.5 with Pandas 1.5.3.
`install_sttts_requirements.sh` uses the Tsinghua PyPI mirror by default. It
preinstalls NumPy and Cython, then builds ESPnet's `pyworld` and
`ctc-segmentation` dependencies without pip build isolation. This avoids a
second, often slow download at the `Installing build dependencies` step. To
use the Aliyun mirror instead:

```bash
STTTS_PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
  bash install_sttts_requirements.sh
```

## 3. Obtain and prepare STTTS models

The public release URLs are:

```bash
mkdir -p model_archives
cd model_archives
wget -c https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/anonymization.zip
wget -c https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/asr.zip
wget -c https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/tts.zip
cd ..
```

Prepare and verify them:

```bash
python prepare_models.py \
  --archives-dir model_archives \
  --output-dir models/voicepat-v2
```

Do not run three independent `unzip` commands unless only inspecting the
archives. `prepare_models.py` checks expected byte sizes and SHA256, verifies
ZIP CRCs, blocks unsafe paths, extracts all three into one directory, and
writes `model_manifest.json`. Expected top-level layout:

```text
models/voicepat-v2/
  model_manifest.json
  anonymization/gan_style-embed/style-embed_wgan.pt
  asr/asr_branchformer_tts-phn_en.zip
  tts/Aligner/aligner.pt
  tts/Embedding/embedding_function.pt
  tts/FastSpeech2_Multi/prosody_cloning.pt
  tts/HiFiGAN_combined/best.pt
```

The ZIP inside `asr/` is ESPnet's model package. Leave it intact; ESPnet opens
and caches it on first model load.

Expected archive SHA256 values from the tested download on 2026-08-08:

```text
0e662a713fb3f01b25000b7d1fa65ff5dd58a39ed7922703df26c8b7a9ba7e32  anonymization.zip
14dd433addca14a57a67d1ad838c830ea1a5cb4070dc6a5be95b14ab7a86133d  asr.zip
c9a40069693d98c4d7a9c17646aa670ded668c0f0cc376175fc26d4fb706b8c1  tts.zip
```

The release publisher does not provide upstream checksums; these hashes are
recorded fingerprints of the files actually tested here.

### If the extracted model directory was uploaded directly

Uploading the already extracted `anonymization/`, `asr/`, and `tts/`
directories is supported. It bypasses `prepare_models.py`, so also upload the
trusted manifest generated on the local machine:

```bash
# Run on the local Mac from the X-VC2 workspace.
scp artifacts/voicepat-v2-models/extracted/model_manifest.json \
  qixiang.xu@SERVER:~/voiceprivacy_sa_deployment/models/voicepat-v2/
```

Do not generate a new manifest from the uploaded files: that would record a
damaged upload as the expected state. The trusted local manifest validates the
six required model files against their original sizes and SHA256 values.

The directory named `asr/ff0250f9303ce1a17a4a472b9b1b295d/` is an ESPnet
download/extraction cache, not one of the six required model files. If it was
uploaded without `config.yaml` and `valid.acc.ave_10best.pth`, move it aside so
ESPnet can rebuild it from the intact inner ZIP:

```bash
cd ~/voiceprivacy_sa_deployment

stat -c '%s %n' models/voicepat-v2/asr/asr_branchformer_tts-phn_en.zip
sha256sum models/voicepat-v2/asr/asr_branchformer_tts-phn_en.zip

mv models/voicepat-v2/asr/ff0250f9303ce1a17a4a472b9b1b295d \
  models/voicepat-v2/asr/ff0250f9303ce1a17a4a472b9b1b295d.partial-upload
```

The expected inner ZIP result is:

```text
bytes=438409407
sha256=1b020d56dc52d66530dc56fcaeb33a28ab989a2ee92a4bcc315076bf70a5154a
```

Keeping the renamed cache makes this operation recoverable. Remove it only
after the first STTTS model load succeeds.

## 4. Verify before inference

```bash
python verify_sa_setup.py \
  --backend sttts \
  --models-dir models/voicepat-v2
```

Expected final line:

```text
setup_verification=PASS
```

Validate an input and all paths without loading models:

```bash
python anonymize_sa.py \
  --audio /path/input.wav \
  --utterance-id smoke-en \
  --speaker-id speaker-smoke \
  --backend sttts \
  --models-dir models/voicepat-v2 \
  --output-dir runs/sttts-check \
  --gpus 0 \
  --check-only
```

## 5. Run one file

One WAV on GPU 0:

```bash
python -u anonymize_sa.py \
  --audio /path/input.wav \
  --utterance-id utt-001 \
  --speaker-id speaker-001 \
  --backend sttts \
  --models-dir models/voicepat-v2 \
  --output-dir runs/sttts \
  --gpus 0
```

## 6. Generate a batch dataset

First build a manifest. For LibriTTS, point `--audio-root` at one split and use
speaker depth 2 because its layout is `speaker/chapter/audio.wav`:

```bash
mkdir -p manifests runs

python prepare_audio_manifest.py \
  --audio-root /data/LibriTTS/train-clean-100 \
  --speaker-parent-depth 2 \
  --extensions .wav \
  --output manifests/train-clean-100.jsonl

wc -l manifests/train-clean-100.jsonl
head -2 manifests/train-clean-100.jsonl
```

For a generic layout such as `/data/source/speaker_id/audio.wav`, use
`--speaker-parent-depth 1`. Utterance IDs are derived from the complete relative
path, so repeated filenames in different directories remain distinct.

Before a large run, create a small deterministic smoke manifest:

```bash
python prepare_audio_manifest.py \
  --audio-root /data/LibriTTS/train-clean-100 \
  --speaker-parent-depth 2 \
  --extensions .wav \
  --limit 20 \
  --output manifests/train-clean-100-smoke20.jsonl
```

Validate all paths and model files without inference:

```bash
python anonymize_sa.py \
  --manifest manifests/train-clean-100.jsonl \
  --backend sttts \
  --models-dir models/voicepat-v2 \
  --output-dir runs/train-clean-100-sttts \
  --anonymization-level spk \
  --gpus 0 \
  --check-only
```

Then run the complete batch on GPU 0:

```bash
python -u anonymize_sa.py \
  --manifest manifests/train-clean-100.jsonl \
  --backend sttts \
  --models-dir models/voicepat-v2 \
  --output-dir runs/train-clean-100-sttts \
  --anonymization-level spk \
  --gpus 0 \
  2>&1 | tee runs/train-clean-100-sttts.log
```

Or generate the McAdams comparison with eight CPU workers:

```bash
python -u anonymize_sa.py \
  --manifest manifests/train-clean-100.jsonl \
  --backend mcadams \
  --output-dir runs/train-clean-100-mcadams \
  --anonymization-level spk \
  --jobs 8
```

Success is indicated by `anonymization=PASS`. The authoritative output index is
`<output-dir>/pairs.jsonl`.

Each row of `pairs.jsonl` directly associates:

```text
original_audio_path
normalized_source_audio_path
anonymized_audio_path
speaker_id
utterance_id
```

Use `normalized_source_audio_path` and `anonymized_audio_path` as the actual
16 kHz source/SA training pair. STTTS resynthesis can change duration, so do not
assume equal sample counts.

If a job stops, rerun the exact same manifest, settings, and output directory;
the upstream pipeline reuses saved intermediate results. A changed manifest or
setting requires a new output directory. If manually splitting a large corpus,
keep all utterances from one speaker in the same shard when speaker-level
pseudonym consistency matters.

## 7. Offline and runtime notes

- All inference checkpoints are local after model preparation.
- Silero VAD is loaded from the installed package; the wrapper prevents the
  upstream `torch.hub` network request.
- `espeak-ng` must provide both its executable and shared library. The verifier
  checks both.
- Some ESPnet imports may request NLTK `cmudict` and
  `averaged_perceptron_tagger`. This STTTS path uses phone ASR and eSpeak, so
  those resources were not required by the verified English smoke test. They
  can be installed once to suppress the messages:

```bash
python -m nltk.downloader -d "$CONDA_PREFIX/nltk_data" \
  averaged_perceptron_tagger cmudict
export NLTK_DATA="$CONDA_PREFIX/nltk_data"
```

- The three archives occupy about 1.16 GB; extracted models occupy about
  1.6 GB. Reserve additional space for intermediate ASR, embeddings, prosody,
  and generated audio.
- The first run creates and caches 5,000 GAN speaker vectors under the output
  directory. Reusing the same output directory reuses intermediates.
- CPU inference is supported but intended only for smoke testing. GPU is the
  recommended deployment.

## 8. Tests

Fast tests (STTTS is skipped unless explicitly enabled):

```bash
python -m unittest discover -s tests -v
```

Full neural smoke test:

```bash
SA_STTTS_MODELS_DIR="$PWD/models/voicepat-v2" \
SA_STTTS_TEST_AUDIO=/path/english.wav \
python -m unittest discover -s tests -p 'test_sttts_pipeline.py' -v
```
