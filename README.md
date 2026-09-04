# X-VC2 Large Streaming Codec

从随机参数训练的大容量连续 latent Codec。它接收冻结 Large Student 的 768-d 50 Hz hidden，
并把语音表示拆为 `Z_inv`、`Z_dyn` 和 `Z_edit`。

## 模型合同

```text
Student hidden 768 -> Keep Head 512 -> Z_inv 128 + Z_dyn 32
waveform -> Causal Acoustic Encoder -> Z_edit 96
[Z_inv, Z_dyn, Z_edit] -> Fusion 640 -> Causal Decoder -> waveform
```

Decoder 是 unconditional 的，不接收 target speaker embedding。未来 speaker condition 只允许
进入独立 Latent Converter。

默认参数量：

```text
Keep Head          746,668
Acoustic Encoder 10,549,536
Fusion            13,348,800
Decoder           39,336,833
Acoustic total    63,235,169
Model total       63,981,837
```

所有模块均为 fresh initialization；训练入口没有加载旧 Base Codec、GAN 或 Keep Adapter
checkpoint 的参数迁移路径。

## 安装与 Smoke

```bash
python -m pip install -e '.[dev]'
xvc2-codec-smoke
xvc2-codec-smoke --config configs/codec_63m.yaml
```

在已有 `ctc-gop` 环境中先运行：

```bash
xvc2-codec-env-check --require-cuda
```

只有输出 `ctc_gop_codec_environment=PASS` 才可直接复用。该检查覆盖 Torch、torchaudio、
PyYAML、二进制版本匹配、STFT API 和 CUDA 可见性。

## Manifest

Source manifest 每行：

```json
{
  "audio_path": "/absolute/path/audio.wav",
  "student_hidden_path": "/absolute/path/student_hidden.pt",
  "speaker_target_path": "/absolute/path/speaker_embedding.pt",
  "phone_target_path": "/optional/path/phone_logits.pt",
  "dyn_target_path": "/optional/path/dyn_anchor.pt",
  "prosody_target_path": "/optional/path/prosody.pt"
}
```

Pair manifest 每行：

```json
{"source": {"audio_path": "...", "student_hidden_path": "...", "speaker_target_path": "..."}, "sa": {"audio_path": "...", "student_hidden_path": "...", "speaker_target_path": "..."}}
```

`student_hidden.pt` 可以直接存 Tensor，或使用 `{"student_hidden": Tensor}`。其它缓存同理。
`speaker_target_path` 是必需项；phone/dyn/prosody anchor cache 是可选项。

正式训练前执行真实 cache 审计：

```bash
xvc2-codec-audit \
  --config configs/codec_63m.yaml \
  --source-manifest /path/codec_source_train_manifest.jsonl \
  --pair-manifest /path/sa_pairs_manifest.jsonl \
  --speaker-target-dim 256
```

它会读取音频和 Tensor，验证 `[T,768]` Student cache、speaker target、可选 anchor
维度、音频/cache 50 Hz 对齐以及每个 pair 的 source/SA 完整性。

## 资源 Benchmark

```bash
CUDA_VISIBLE_DEVICES=0 xvc2-codec-benchmark \
  --config configs/codec_63m.yaml --batch-size 1 --audio-seconds 3.2

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m xvc2_codec.benchmark \
  --config configs/codec_63m.yaml --batch-size 1 --audio-seconds 3.2
```

默认同时执行 Generator reconstruction/GAN/FM 和 Discriminator 更新，输出 step time、
global audio seconds/second 与每 rank 峰值显存。`--no-with-discriminator` 可单独测 warm-up。

## 统一训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m xvc2_codec.train \
  --config configs/codec_63m.yaml \
  --source-manifest /path/codec_source_train_manifest.jsonl \
  --pair-manifest /path/sa_pairs_manifest.jsonl \
  --output-dir runs/codec-63m-v1 \
  --speaker-target-dim 256 \
  --batch-size 1 \
  --segment-seconds 3.2 \
  --pair-probability 0.15
```

训练日程自动来自配置：

```text
0-10k       reconstruction warm-up
10k-30k     complex MS-STFT GAN/FM ramp
30k-60k     SA Z_inv/Z_dyn loss ramp
60k-end     long joint training + EMA
```

Checkpoint 保存 generator、训练期 style head、discriminator、两个 optimizer、EMA 和 RNG。
每个 step 的 source/pair 选择和 per-rank 样本索引由固定 seed 与 global step 推导，因此 resume
不会依赖不可见的 DataLoader shuffle position。

## 当前边界

- 连续 latent；没有 RVQ/VQ。
- 没有 Converter、GRL、cycle loss、source-speaker repel 或 target acoustic memory。
- 默认只有 complex multi-scale STFT discriminator，没有 MPD。
- 正式大训练前仍需服务器 2-step DDP smoke、显存/吞吐 benchmark、32-item overfit 和
  full/chunk/reset/flush acceptance。
