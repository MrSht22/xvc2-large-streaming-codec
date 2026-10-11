# X-VC 配对数据接入

`xvc2_codec.vc_data.VCPairDataset` 直接读取 X-VC 原始 JSONL：

```json
{"index":0,"source_utt":"utt_00295279_01","source_wav_path":"/data/generated.flac","target_utt":"utt_00295279_11","target_wav_path":"/data/natural.mp3"}
```

`source_utt`、`index` 可选；路径和 `target_utt` 必填。不从 utterance 后缀推断身份，
不要求 SA、Student cache 或额外 reference manifest。这里是后续 VC 的数据接口，
Codec 续训入口 `xvc2_codec.train` 仍使用已有 source/SA 数据。流式 Converter 和配对前向接口
见 [VC_STREAMING.md](VC_STREAMING.md)；两阶段 VC trainer 和 ERes2Net/Mel 条件接入
见 [VC_TRAINING.md](VC_TRAINING.md)。

## source / target 使用方式

复用 `X-VC/models/codec/sac/dataloader.py` 的角色分配；默认配置见
`configs/vc_pairs.yaml`。

| role_assignment_mode | 输入 source | 监督 target | 默认训练比例 |
| --- | --- | --- | --- |
| standard | generated source | natural target | 40% |
| reconstruction | natural target | 同一 natural target | 20% |
| reversed | natural target | generated source | 40% |

角色分配发生在音频读取之前，路径、utterance ID、条件音频一起切换。
保留原 JSONL `index` 和 `manifest_target_utt` 便于追踪；不套用 X-VC 的
`target_utt[:-3]` 截断规则。验证集使用 standard 方向、从零开始裁剪，避免随机反向/自重建混入指标。

音频解码复用当前 Codec 的重试/FFmpeg fallback，支持后端可解码的 MP3、FLAC、WAV；
转为 mono（多通道取均值）及 16kHz，复用 X-VC 的音量归一化规则和默认 40Hz 高通。
这里不需要 Whisper token/SSL feature；后续由冻结 Large Student 和 Codec 提取三路 latent。

双方取完整 320-sample 帧的共同有效区间，同起点随机裁剪训练片段（默认 2.4 秒）。
短片段补零到训练长度，验证片段最多 12 秒；batch 可变长部分也补零。
与原 loader 相比，共同长度考虑双方，padding 由真实裁剪长度计算，避免 target 较短时错补或重复补零。

`source_wav_cond` / `target_wav_cond` 来自**完成角色分配后的双方整句共同区间**，
`mask_target_condition: true` 时将当前裁剪区间置零。它们不是同身份异句参考。
短音频若完全被裁剪覆盖，条件会全零，这是该采样规则的实际结果；IO 报告记录
`target_condition_nonzero_samples`。这类样本不能提供有效的目标参考声音。

该 loader 假设配对内容/时间轴已一致：共享裁剪不是时间对齐，也不执行 DTW、延迟估计或音频拉伸。

## 接入 DataLoader

```python
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from xvc2_codec.vc_data import VCPairDataConfig, VCPairDataset

config = VCPairDataConfig(**yaml.safe_load(Path("configs/vc_pairs.yaml").read_text()))
dataset = VCPairDataset(Path("/data/vc_pairs.jsonl"), config, random_crop=True)
loader = DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=0,
    collate_fn=dataset.collate, generator=torch.Generator().manual_seed(42),
)
```

角色抽样及裁剪使用 PyTorch RNG。固定 worker 数量、遍历顺序和随机种子可复现；
`num_workers=0` 时还应在遍历前调用 `torch.manual_seed(42)`。

Batch 接口沿用当前 Codec 的 waveform view：

- `source` / `target`: `waveform [B,1,N]`，`frames [B]` 是补零前有效帧数。
- `natural`: 原 JSONL 自然 target 的共享 crop，不随反向角色切换，用于 Decoder 普通自重建。
- `source_wav_cond` / `target_wav_cond`: `[B,1,N_cond]`。
- `condition_frames [B]`: 条件整句的有效帧数，不包含 batch padding；置零的训练区间仍属于原时间轴。
- `reference_waveform [B,1,N_ref]` / `reference_samples [B]`: 训练使用的参考，长度不含 padding。
- `reference_kind` / `reference_wav_path`: 独立异句或配对条件的逐样本 metadata。
- `role_assignment_mode`、路径、IDs、`start_frame`、原音频 sample counts: 逐样本 metadata。

后续 paired loss 必须使用 `frames` 排除补零帧；提取 Student hidden 后还需按其
真实有效帧数裁剪（400-sample 感受野、320 stride 可能少一帧），不能直接把补零长度当监督长度。
条件编码也须使用 `condition_frames` 排除 batch padding。

X-VC 原模型从 target 训练片段提取 speaker condition，从 masked target 整句提取 Mel condition。
当前 VC trainer 从同一个 reference 提取 speaker embedding 与 Mel；可显式提供
`target_reference_wav_path` / `source_reference_wav_path`，并随角色切换。缺少字段时使用
分配后的 target 剩余音频（移除监督 crop），保持配对条件基线；不足 0.5 秒或静音时训练会报错，
需补独立 reference。完整参考定义与方向见 [VC_TRAINING.md](VC_TRAINING.md)。

## 检查原始 JSONL 的 IO

### 从一份大 manifest 划分 train / val

在服务器 Codec checkout 中运行；target 自然音频目录必须在当前机器可访问：

```bash
python src/xvc2_codec/vc_split.py \
  --manifest /path/to/all_pairs.jsonl \
  --train-manifest /path/to/vc_train.jsonl \
  --val-manifest /path/to/vc_val.jsonl \
  --val-size 256 --seed 42
```

也可使用 `PYTHONPATH=src python -m xvc2_codec.vc_split`。直接运行上述脚本只需 Python 标准库，
不加载模型、不解码音频、不复制音频文件；两遍流式读取 JSONL，只保存分组索引，适用于大 manifest。

`256` 指 **256 个不同的自然 `target_wav_path`**。它们的全部生成变体都进入 val，
其余行全部进入 train；因此 val 行数可能超过 256。原字段/index 保留，不重新编号。
固定 seed 和目录内容时划分及参考选择可复现；输出已经存在时会报错，保护已有文件。

每个 val target 从其音频所在目录选另一句 reference，并写入 `target_reference_wav_path`。
对于 Emilia 的 `.../EN_B00006_S08734/mp3/xxx.mp3`，只查同一 `mp3` 目录，不搜索语言根目录。
该规则假定同一音频目录属于同一说话人；排除当前 target 的同名不同编码副本和该组 source。
找不到 reference 或 target 不存在的组留在 train，继续选择其他组；不足 256 个可选组时，
在创建输出前报错。脚本输出实际 train/val target 数和行数，以及尝试期间跳过的组数。

这是按 target utterance 划分，speaker 可以同时出现在 train/val。
同目录 reference 也可能是 train 中其他句子的自然音频；当前规则不要求独立的 reference pool。
脚本只检查路径，不验证参考时长/静音或音频内容；已有训练 frontend 仍要求至少 0.5 秒非静音参考。

训练时将这两个输出分别传给 `--train-manifest` / `--val-manifest`。
要每次评估全部 val 行，应将 `--val-pairs` 设置为脚本报告的 `val_rows`；默认只评估前 16 行。

### 抽样检查读取和裁剪

```bash
python -m xvc2_codec.vc_data \
  --manifest /data/vc_pairs.jsonl \
  --config configs/vc_pairs.yaml \
  --mode train --max-pairs 16 --batch-size 4 --seed 42 \
  --output runs/vc-pair-io/train.json
```

改成 `--mode val` 可以检查固定正向输入。命令只加载选中音频并记录角色、路径、
裁剪位置、原始长度、有效帧数、条件非零 sample 数及 reference 类型/长度；不加载模型或启动训练。
IO 的 PASS 表示读取/形状检查通过，不能代替 reference 质量或时间对齐检查。
小样本角色计数只是实际抽样结果，不要求刚好符合 40/20/40。
