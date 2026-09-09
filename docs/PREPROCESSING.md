# Codec Preprocessing Runbook

该流程从 source audio selection manifest 和 VoicePrivacy SA `pairs.jsonl` 生成正式训练 cache。
Student temporal cache 使用连续 FP16 binary shard；GST speaker target 使用共享矩阵和 row index。

## 0. 路径

在服务器新终端执行：

```bash
conda activate ctc-gop

export ROOT=/inspire/hdd2/project/multilingualspeechrecognition/chenxie-25019/qixiangxu/X-VC2
export CODEC=$ROOT/xvc2-large-streaming-codec
export STUDENT=$ROOT/xvc2-large-streaming-student
export SOURCE=$ROOT/xvc2-large-streaming-student/runs/codec-audio-all-cap30h-v1/train_audio.jsonl
export PAIR=$CODEC/sa_generation/runs/sa-librispeech-clean300h-sttts-spk-w16/pairs.jsonl
export STUDENT_CKPT=$STUDENT/runs/student-12x768-2500h-extension-1epoch-v1/step-053168.pt
export SA_RUN=$CODEC/sa_generation/runs/sa-librispeech-clean300h-sttts-spk-w16
export MODELS=$CODEC/sa_generation/models/voicepat-v2
export VENDOR=$CODEC/sa_generation/third_party/Voice-Privacy-Challenge-2026
export PREP=$CODEC/runs/codec-preprocessing-v1

cd "$CODEC"
git pull --ff-only origin main
python -m pip install -e '.[dev]'
mkdir -p "$PREP/logs"
```

## 1. Plan 和磁盘门槛

```bash
PYTHONPATH=src \
python -m xvc2_codec.preprocess plan \
  --source-manifest "$SOURCE" \
  --pair-manifest "$PAIR" \
  --output-dir "$PREP/plan" \
  --references-per-speaker 3 \
  2>&1 | tee "$PREP/logs/plan.log"
```

目的：

- 生成所有 source、pair-source、pair-SA view 的确定性 `inventory.jsonl`；
- 每个 source speaker 选择最多 3 条 GST reference；
- 估计 FP16 `[T,768]` hidden 与 `[T,40]` phone-logit cache 空间；
- 检查目标文件系统可用空间，并预留 10% 余量。
- 报告 original/SA 的逐条 duration drift 分位数，作为同帧 SA loss 的对齐诊断。

继续条件：终端末行必须是：

```text
codec_preprocess_plan=PASS
```

若为 `FAIL`，不要执行下一阶段；先更换 `PREP` 到有足够空间的文件系统。`plan` 不解码音频、
不运行 Student，也不修改原数据。

如果 `pair_duration_alignment.status=NEEDS_ATTENTION`，先运行冻结 Student 对齐 probe：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$CODEC/src:$STUDENT/src" \
python -m xvc2_codec.preprocess probe-pair-alignment \
  --pair-manifest "$PAIR" \
  --student-checkpoint "$STUDENT_CKPT" \
  --output "$PREP/plan/pair-alignment-probe.json" \
  --device cuda:0 \
  --max-items 256 \
  --max-seconds 20 \
  --max-lag-frames 30 \
  2>&1 | tee "$PREP/logs/pair-alignment-probe.log"
```

该 probe 检查 Student hidden 的 zero-lag cosine、最佳 temporal lag、phone argmax 一致率和
错配 pair baseline。只有 `codec_pair_alignment_probe=PASS` 才可直接使用共享 frame crop；否则
先修正 pair 对齐，不得启动全量 cache 提取。

## 2. 四卡提取 Student cache

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH="$CODEC/src:$STUDENT/src" \
torchrun --standalone --nproc_per_node=4 \
  -m xvc2_codec.preprocess extract-student \
  --inventory "$PREP/plan/inventory.jsonl" \
  --student-checkpoint "$STUDENT_CKPT" \
  --output-dir "$PREP/student" \
  --batch-size 8 \
  --max-batch-seconds 240 \
  --num-workers 2 \
  --shard-max-frames 500000 \
  --require-cuda \
  2>&1 | tee "$PREP/logs/extract-student.log"
```

每个 rank 独立处理 `inventory[rank::4]`。单个 shard 约 0.75 GiB hidden 加 0.04 GiB logits；
完成标记使用 fingerprint，重复执行会打印 `SKIP_COMPLETE`，中断后可直接重跑同一命令。
`--max-batch-seconds` 限制 padding 后一个 batch 的总音频长度，避免少数 120 秒音频导致 OOM。

继续条件：四个 rank 都打印：

```text
codec_student_extraction=PASS
```

## 3. GST speaker target

该阶段必须切换到生成 SA 时使用的 `voiceprivacy-sa` 环境：

```bash
conda activate voiceprivacy-sa
cd "$CODEC"

PYTHONPATH=src \
python -m xvc2_codec.preprocess extract-speakers \
  --reference-dir "$PREP/plan/speaker_references" \
  --models-dir "$MODELS" \
  --vendor-dir "$VENDOR" \
  --sa-run-dir "$SA_RUN" \
  --pair-manifest "$PAIR" \
  --output-dir "$PREP/speakers" \
  --devices cuda:0 \
  2>&1 | tee "$PREP/logs/extract-speakers.log"
```

这一步只对每个 source speaker 的最多 3 条 reference 提取 GST 并平均到 speaker level，同时从
已有 SA run 的 STTTS intermediate store 收集匿名 WGAN speaker vectors。不会重新生成 SA 音频。

继续条件：

```text
codec_speaker_extraction=PASS
```

并且实际 `embedding_dim=128`。

## 4. Finalize 和 audit

```bash
conda activate ctc-gop
cd "$CODEC"

PYTHONPATH=src \
python -m xvc2_codec.preprocess finalize \
  --source-manifest "$SOURCE" \
  --pair-manifest "$PAIR" \
  --student-cache-dir "$PREP/student" \
  --speaker-cache-dir "$PREP/speakers" \
  --output-dir "$PREP/manifests" \
  2>&1 | tee "$PREP/logs/finalize.log"

PYTHONPATH=src \
python -m xvc2_codec.audit \
  --config configs/codec_63m.yaml \
  --source-manifest "$PREP/manifests/source_train_cache.jsonl" \
  --pair-manifest "$PREP/manifests/pair_train_cache.jsonl" \
  --speaker-target-dim 128 \
  --max-items 1000 \
  2>&1 | tee "$PREP/logs/audit-1000.log"
```

先要求：

```text
codec_preprocess_finalize=PASS
```

再要求 1000 条抽查输出：

```text
codec_manifest_audit=PASS
```

抽查通过后，去掉 `--max-items 1000` 做全量 manifest audit，再进入 Codec benchmark 和短训练。
`dyn_target_path` 与 `prosody_target_path` 当前不生成，因此对应可选 anchor loss 为 0；没有使用伪造
target 填充它们。
