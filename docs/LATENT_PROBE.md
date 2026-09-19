# Latent Probe and Leakage Analysis

This tool analyzes a **frozen** Codec checkpoint. It does not update Codec
weights and it does not train a Converter.

The entry point is:

```text
xvc2_codec.latent_probe
```

It runs three analyses:

1. **Content probe**: a linear frame classifier predicts the cached phone target
   from each of `Z_inv`, `Z_dyn`, and `Z_edit`. Higher accuracy means more
   recoverable phone information.
2. **Speaker leakage probe**: a linear utterance-level classifier predicts the
   manifest `speaker_id` from the mean of each latent. Higher accuracy means
   more recoverable source-speaker identity, so this is leakage for `Z_inv`.
3. **Pair consistency**: on aligned source/SA pairs, it reports framewise
   cosine similarity and MSE for each latent. `Z_inv` should have high cosine
   and low MSE; `Z_dyn` and `Z_edit` are diagnostic only.

## Data contract

Use a held-out source manifest and a held-out pair manifest whenever possible.
Do not use the Codec training manifests as the final result. The source rows
should contain:

```text
utterance_id
speaker_id
audio_path
student_hidden_path (or shard fields)
phone_target_path (or shard fields)
```

The pair rows should use the finalized source/SA cache schema and contain the
same alignment contract used by training. The pair manifest's `speaker_id` is
the original source speaker label. It is not automatically the anonymous SA
identity.

The phone probe uses the cached phone target's argmax as a frame label. It is a
diagnostic probe, not a full ASR evaluation. The speaker probe uses one fixed
random crop per utterance and mean-pools the latent over time. Train/validation
splits are made by utterance id, so frames from the same utterance do not cross
the split.

## Build two evaluation caches

The cache builder only selects rows and writes new JSONL manifests. It does not
copy audio, Student hidden shards, or phone caches. The candidate pool must
already be a held-out finalized cache; do not pass the Codec training manifest
as both the candidate pool and an exclusion manifest.

The builder creates:

```text
<output>/content/source_eval_cache.jsonl
<output>/content/pair_eval_cache.jsonl
<output>/leakage/source_eval_cache.jsonl
<output>/leakage/pair_eval_cache.jsonl
<output>/selection_report.json
```

The content set is speaker-balanced, phone-labeled, and defaults to 128
speakers with 4 utterances per speaker. The leakage set defaults to 64 speakers
with 8 utterances per speaker and selects across distinct chapter ids whenever
the manifest exposes LibriSpeech-style utterance ids. A speaker must have
enough rows in the candidate pool or the command fails instead of silently
producing an invalid leakage set.

On the server, after pushing the cache-builder commit:

```bash
conda activate ctc-gop
cd "$CODEC"
export PYTHONPATH=src

# These must be held-out finalized caches, not the Codec training manifests.
SOURCE_POOL="$HELDOUT_PREP/manifests/source_eval_pool_cache.jsonl"
PAIR_POOL="$HELDOUT_PREP/manifests/pair_eval_pool_cache.jsonl"
OUT="$HELDOUT_PREP/manifests/latent-probe-eval-v2"

mkdir -p "$OUT"
PYTHONPATH=src \
python -m xvc2_codec.build_eval_cache \
  --source-manifest "$SOURCE_POOL" \
  --pair-manifest "$PAIR_POOL" \
  --output-dir "$OUT" \
  --content-speakers 128 \
  --content-utterances-per-speaker 4 \
  --leakage-speakers 64 \
  --leakage-utterances-per-speaker 8 \
  --seed 1 \
  2>&1 | tee "$OUT/build.log"
```

If the candidate pool was assembled from a larger source inventory that also
contains training rows, pass every training manifest to `--exclude-manifest`
before selection:

```bash
  --exclude-manifest "$PREP/manifests/source_train_cache.jsonl" \
  --exclude-manifest "$PREP/manifests/pair_train_cache.jsonl"
```

The builder excludes matching utterance ids and audio paths. It also checks
that selected source rows have matching pair rows. The output
`selection_report.json` is the first gate: confirm the requested speaker and
utterance counts before running the latent probe.

For the subsequent probes, use the two directories independently:

```bash
SOURCE_EVAL="$OUT/content/source_eval_cache.jsonl"
PAIR_EVAL="$OUT/content/pair_eval_cache.jsonl"
# Run the content probe with these paths.

SOURCE_EVAL="$OUT/leakage/source_eval_cache.jsonl"
PAIR_EVAL="$OUT/leakage/pair_eval_cache.jsonl"
# Run the leakage probe with these paths.
```

The current latent probe reports both content and speaker probes in one run;
the separate manifests are intentional. Content analysis needs broad phone and
speaker coverage, while leakage analysis needs repeated utterances per speaker
for a meaningful utterance-disjoint validation split.

## Server command

On the server, after pushing commit `9782f60`:

```bash
conda activate ctc-gop
cd "$CODEC"
export PYTHONPATH=src

CKPT="$RUNS/codec-63m-v1/step-300000.pt"
SOURCE_EVAL="$PREP/manifests/source_eval_cache.jsonl"
PAIR_EVAL="$PREP/manifests/pair_eval_cache.jsonl"
OUT="$PREP/latent-probe/step-300000"

mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src \
python -m xvc2_codec.latent_probe \
  --config configs/codec_63m.yaml \
  --checkpoint "$CKPT" \
  --source-manifest "$SOURCE_EVAL" \
  --pair-manifest "$PAIR_EVAL" \
  --output-dir "$OUT" \
  --weights ema \
  --device cuda:0 \
  --segment-seconds 3.2 \
  --max-source-items 512 \
  --max-pair-items 256 \
  --extract-batch-size 8 \
  --max-frames 200000 \
  --epochs 20 \
  --probe-batch-size 4096 \
  --seed 1 \
  2>&1 | tee "$OUT/run.log"
```

This first implementation is single-GPU and should **not** be launched with
`torchrun`. The model is frozen and the probe heads are small; one GPU avoids
duplicating the full Codec across DDP ranks. `--extract-batch-size` can be
lowered if the selected checkpoint or audio crop uses more memory.

For an exploratory run on the existing training manifests, change only the
two manifest paths and label the output as training-set diagnostics. Do not
present those numbers as held-out leakage results.

## Outputs

The command writes:

```text
report.json
probe_metrics.csv
run.log
```

The report contains:

```text
content_probe.z_inv / z_dyn / z_edit
speaker_leakage_probe.z_inv / z_dyn / z_edit
pair_consistency.mean.z_inv_mse
pair_consistency.mean.z_inv_cosine
pair_consistency.mean.z_dyn_cosine
pair_consistency.mean.z_edit_cosine
```

The content and leakage rows include the validation accuracy and a majority
class baseline. Compare latent names on the same manifest and seed; do not
interpret an absolute accuracy without the baseline and class count.

Optional latent dumps can be enabled with:

```bash
  --save-latents
```

This writes `source_latents.pt` and can be large. Keep it for follow-up plots
or additional probes only when the storage budget permits.

## Reading the result

The intended pattern is:

```text
Z_inv:  high phone accuracy, low speaker leakage, high source/SA cosine
Z_dyn:  lower phone accuracy, some speaker/prosody information
Z_edit: speaker/style information, possibly some residual phone leakage
```

This pattern is a strong indication that a later Converter can focus on
`Z_dyn` and `Z_edit`. It is not, by itself, proof of successful VC. A stricter
follow-up should use speaker-disjoint evaluation and compare against a
same-size random or majority baseline.
