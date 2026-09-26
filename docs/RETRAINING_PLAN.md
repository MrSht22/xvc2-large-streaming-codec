# Codec Disentanglement and Quality Retraining Plan

日期：2026-09-26。

## 目标与边界

本轮不从 step 0 重训。以旧版 step 60k 为共同初始化点，保留已经形成的重建、GAN 和
`Z_inv` phone 能力，重新训练 60k 之后的表示分工：

- `Z_inv` 保留 phoneme/content；
- `Z_dyn` 显式承载相对 F0、voicing、相对能量和局部 F0 变化；
- `Z_edit` 保留不应进入 content/dynamic 的 speaker 与 acoustic detail；
- SA pair 继续约束 `Z_inv`，但只弱约束 `Z_dyn`，不再要求动态轨迹近似逐元素相等；
- PESQ/STOI 的最后恢复由 Decoder-only quality stage 完成，不回写 latent encoder。

不增加 F0-GRL。F0 是未来 Converter 需要学习修改的 dynamic 因素，不应从 `Z_dyn` 和
`Z_edit` 中同时删除。

## Joint loss

Joint 阶段保持现有 reconstruction、complex multi-scale STFT adversarial、feature matching、
style 和 SA loss，并增加：

```text
L_dyn = 0.10 L_f0 + 0.05 L_vuv + 0.10 L_energy + 0.05 L_delta_f0
L_phone_adv = 0.02 L_phone_GRL(Z_dyn) + 0.05 L_phone_GRL(Z_edit)
```

其中 prosody target 是 50 Hz 四维离线 cache：

```text
[utterance-normalized log-F0, V/UV,
 utterance-normalized log-RMS energy, delta normalized log-F0]
```

`L_f0` 只在 voiced frame 上计算 SmoothL1；`L_vuv` 使用 BCEWithLogits；`L_energy` 在有效
frame 上计算 SmoothL1；`L_delta_f0` 只在前后两帧都 voiced 时计算 SmoothL1。Phone adversary
使用 Student soft phone posterior 的 KL target；classifier 正常学习，GRL 对 encoder 梯度反向。
默认 GRL scale 为 `Z_dyn=0.05`、`Z_edit=0.10`。

SA loss 保持：

```text
L_SA = 1.0 L_inv_smooth_l1 + 0.01 L_dyn_trajectory_correlation
```

`L_dyn` 和 Phone-GRL 从初始化 checkpoint 的 global step 开始，用 10k step 线性 ramp。旧
step 60k 的 optimizer、scaler 和 RNG 不恢复；模型、discriminator 与兼容 EMA 权重会加载，新增
adversary 随机初始化。

## Stage A: 60k diagnostic fork

从旧 step 60k 启动 `joint`，先只跑 10k 到 70k。继续长训前同时检查：

1. reconstruction、WER、SIM 没有突变；
2. `Z_inv` content probe 不明显下降；
3. `Z_dyn` 对 F0/VUV/energy/delta 的 held-out probe 优于旧 60k；
4. 独立 phone probe 在 `Z_dyn`/`Z_edit` 上下降，而 `Z_inv` 保持；
5. source/SA 的 `Z_inv` consistency 保持，`Z_dyn` 不出现方差塌缩。

如果第 3 项没有改善，先调整 prosody 权重；如果第 4 项没有改善，再调整 GRL，不应直接同时放大
所有 loss。如果 WER/SIM 或 reconstruction 明显恶化，停止该 fork，不进入长训。

## Stage B: joint continuation

70k gate 通过后恢复同一个 run，先到 80k 再复核一次，随后继续到 300k。checkpoint 必须用
`--resume`，因为该路径需要恢复 optimizer、scaler、RNG、mode 和 `phase_start_step`。

模型选择不只看训练 objective。固定 eval cache 上同时保留：

- source full reconstruction: WER, SIM, STOI, PESQ-NB/WB；
- SA full reconstruction；
- latent content/speaker/prosody probes；
- source-SA consistency 与 latent variance。

## Stage C: Decoder-only quality refinement

从选中的 late joint checkpoint 用 `configs/codec_63m_quality.yaml` 和 `--mode quality` 新建 run。
该模式冻结 Keep Head、Acoustic Encoder、Fusion、style head 和 phone adversaries，只训练 Decoder；
Discriminator 保持训练。目标仅包含 reconstruction、MS-STFT adversarial 和 feature matching。

quality run 默认 300k 到 330k、Generator LR `2e-5`、Discriminator LR `1e-4`，并建议
`--pair-probability 0`，只用原始高质量 source audio 恢复波形细节。最终 checkpoint 按
STOI/PESQ 的 Pareto 改善选择，同时要求 WER、SIM 和 latent probe 不因 Decoder 精修发生回退。

## Checkpoint semantics

- `--initialize-from`: 只验证 model/discriminator architecture，加载兼容权重，创建新 optimizer、
  scaler 与随机状态；用于旧 60k 分叉和 quality 新阶段。
- `--resume`: 要求 config 与 mode 完全一致，恢复 optimizer、scaler、EMA、RNG 和 ramp 起点；用于
  同一阶段中断续训。
- 两个参数互斥。旧 checkpoint 只允许缺失本次新增的两个 phone adversary。

## Server execution

新终端先定义全部路径。`OLD_60K` 需要替换为旧 run 的真实 checkpoint：

```bash
conda activate ctc-gop

export ROOT=/inspire/hdd2/project/multilingualspeechrecognition/chenxie-25019/qixiangxu/X-VC2
export CODEC=$ROOT/xvc2-large-streaming-codec
export STUDENT=$ROOT/xvc2-large-streaming-student
export SOURCE=$STUDENT/runs/codec-audio-all-cap30h-v1/train_audio.jsonl
export PAIR=$CODEC/sa_generation/runs/sa-librispeech-clean300h-sttts-aligned-transcript-v2/pairs.jsonl
export PREP=$CODEC/runs/codec-preprocessing-aligned-transcript-v2
export SOURCE_CACHE=$PREP/manifests/source_train_cache.jsonl
export PAIR_CACHE=$PREP/manifests/pair_train_cache.jsonl
export OLD_60K=/absolute/path/to/old-codec-run/step-060000.pt
export JOINT_RUN=$CODEC/runs/codec-63m-disentangled-v2
export QUALITY_RUN=$CODEC/runs/codec-63m-disentangled-v2-quality

cd "$CODEC"
git pull --ff-only origin main
python -m pip install -e '.[dev,prosody]'
mkdir -p "$PREP/logs" "$JOINT_RUN/logs" "$QUALITY_RUN/logs"
```

只新增 prosody cache，不重跑 Student 或 speaker extraction：

```bash
PYTHONPATH=src \
torchrun --standalone --nproc_per_node=8 \
  -m xvc2_codec.preprocess extract-prosody \
  --inventory "$PREP/plan/inventory.jsonl" \
  --output-dir "$PREP/prosody" \
  --shard-max-frames 500000 \
  --pitch-floor 50 \
  --pitch-ceiling 600 \
  2>&1 | tee "$PREP/logs/extract-prosody.log"

PYTHONPATH=src \
python -m xvc2_codec.preprocess finalize \
  --source-manifest "$SOURCE" \
  --pair-manifest "$PAIR" \
  --student-cache-dir "$PREP/student" \
  --prosody-cache-dir "$PREP/prosody" \
  --speaker-cache-dir "$PREP/speakers" \
  --alignment-dir "$PREP/alignment" \
  --output-dir "$PREP/manifests" \
  2>&1 | tee "$PREP/logs/finalize-v2.log"

PYTHONPATH=src \
python -m xvc2_codec.audit \
  --config configs/codec_63m.yaml \
  --source-manifest "$SOURCE_CACHE" \
  --pair-manifest "$PAIR_CACHE" \
  --speaker-target-dim 128 \
  2>&1 | tee "$PREP/logs/audit-v2-full.log"
```

60k 到 70k diagnostic fork：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=src \
torchrun --standalone --nproc_per_node=4 \
  -m xvc2_codec.train \
  --config configs/codec_63m.yaml \
  --source-manifest "$SOURCE_CACHE" \
  --pair-manifest "$PAIR_CACHE" \
  --output-dir "$JOINT_RUN" \
  --speaker-target-dim 128 \
  --mode joint \
  --initialize-from "$OLD_60K" \
  --steps 10000 \
  --batch-size 8 \
  --segment-seconds 3.2 \
  --pair-probability 0.15 \
  --num-workers 4 \
  --prefetch-factor 4 \
  2>&1 | tee "$JOINT_RUN/logs/train-060k-070k.log"
```

70k gate 通过后，先续到 80k；再次通过后续到配置上限 300k：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src \
torchrun --standalone --nproc_per_node=4 -m xvc2_codec.train \
  --config configs/codec_63m.yaml \
  --source-manifest "$SOURCE_CACHE" --pair-manifest "$PAIR_CACHE" \
  --output-dir "$JOINT_RUN" --speaker-target-dim 128 --mode joint \
  --resume "$JOINT_RUN/step-070000.pt" --steps 10000 \
  --batch-size 8 --segment-seconds 3.2 --pair-probability 0.15 \
  --num-workers 4 --prefetch-factor 4 \
  2>&1 | tee "$JOINT_RUN/logs/train-070k-080k.log"

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src \
torchrun --standalone --nproc_per_node=4 -m xvc2_codec.train \
  --config configs/codec_63m.yaml \
  --source-manifest "$SOURCE_CACHE" --pair-manifest "$PAIR_CACHE" \
  --output-dir "$JOINT_RUN" --speaker-target-dim 128 --mode joint \
  --resume "$JOINT_RUN/step-080000.pt" \
  --batch-size 8 --segment-seconds 3.2 --pair-probability 0.15 \
  --num-workers 4 --prefetch-factor 4 \
  2>&1 | tee "$JOINT_RUN/logs/train-080k-300k.log"
```

最后从选定 joint checkpoint 开启 source-only Decoder quality refinement：

```bash
export JOINT_FINAL=$JOINT_RUN/step-300000.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src \
torchrun --standalone --nproc_per_node=4 -m xvc2_codec.train \
  --config configs/codec_63m_quality.yaml \
  --source-manifest "$SOURCE_CACHE" --pair-manifest "$PAIR_CACHE" \
  --output-dir "$QUALITY_RUN" --speaker-target-dim 128 --mode quality \
  --initialize-from "$JOINT_FINAL" \
  --batch-size 8 --segment-seconds 3.2 --pair-probability 0 \
  --num-workers 4 --prefetch-factor 4 \
  2>&1 | tee "$QUALITY_RUN/logs/train-quality-300k-330k.log"
```
