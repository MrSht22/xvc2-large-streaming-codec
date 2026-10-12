# 两阶段流式 latent VC

训练入口：`python -m xvc2_codec.vc_train`，配置：`configs/vc_train.yaml`。
Student、KeepHead、AcousticEncoder、Fusion 在两个阶段都冻结，latent teacher 不随 VC 训练改变。
目标 speaker embedding 和 reference Mel 只进入 Converter，Decoder 不接收目标条件。

| 阶段 | 更新模块 | 默认学习率 | 自然音频普通自重建 |
| --- | --- | --- | --- |
| `converter` | Converter，包括 reference encoder/attention | `1e-4` | Decoder 冻结 |
| `decoder` | Converter + 整个 Decoder | `1e-4` / `2e-5` | 每个 batch 都加入 |

第一阶段应先在固定 Decoder 下验证转换收益；不要仅因达到 10k 就切换第二阶段。
第二阶段从该 VC checkpoint 初始化，建立新的 optimizer 和阶段内 step。
`--resume` 才是同阶段续训：恢复 optimizer、AMP scaler、epoch 和 batch cursor。
配置、数据路径、batch size、world size、基座 checkpoint 必须一致。
新 checkpoint 保存每卡已消费样本数；旧 checkpoint 用原 batch size 和 batch cursor 换算。
sampler 从该样本位置开始，不重新读取已经消费的音频。
角色/crop 的随机采样按 epoch/item 确定。
不要训练中切换到新的 Codec。

## 从零训练 Converter 100k（两卡、每卡 batch24）

先在服务器拉取代码，保留原训练环境中的 `XVC_ROOT` 和 `SPEAKER_DIR`。以下沿用原先的
step500000 Codec、Student 和 train/val。Converter/reference encoder/attention 从头训练；
整个 Codec 和 Student 始终冻结，不载入之前的 VC checkpoint。

```bash
export ROOT=/inspire/hdd2/project/multilingualspeechrecognition/chenxie-25019/qixiangxu/X-VC2
export CODEC="$ROOT/xvc2-large-streaming-codec"
export STUDENT="$ROOT/xvc2-large-streaming-student"
export PYTHONPATH="$CODEC/src:$STUDENT/src"
export CUDA_VISIBLE_DEVICES=0,1
export VC_CONFIG="$CODEC/runs/vc-configs/vc_converter_100k_2gpu_b24.yaml"

# XVC_ROOT: 原训练使用的 X-VC checkout；SPEAKER_DIR: 原 ERes2Net 权重目录。
: "${XVC_ROOT:?请先设置为原训练的 X-VC checkout 路径}"
: "${SPEAKER_DIR:?请先设置为原训练的 ERes2Net 权重目录}"

cd "$CODEC"
git pull --ff-only

python - <<'PY'
import os
from pathlib import Path
import yaml

config = yaml.safe_load(Path("configs/vc_train.yaml").read_text())
config["training"].update(max_steps=100000, save_interval=5000)
destination = Path(os.environ["VC_CONFIG"])
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(yaml.safe_dump(config, sort_keys=False))
PY

torchrun --standalone --nproc_per_node=2 -m xvc2_codec.vc_train \
  --config "$VC_CONFIG" \
  --codec-config configs/codec_63m_quality.yaml \
  --codec-checkpoint "$CODEC/runs/codec-63m-disentangled-v2/step-500000.pt" \
  --student-checkpoint "$STUDENT/runs/student-12x768-2500h-extension-1epoch-v1/step-053168.pt" \
  --weights ema \
  --xvc-root "$XVC_ROOT" --speaker-model-dir "$SPEAKER_DIR" \
  --train-manifest "$CODEC/runs/vc-data/train.jsonl" \
  --val-manifest "$CODEC/runs/vc-data/val.jsonl" \
  --stage converter \
  --batch-size 24 --steps 100000 \
  --num-workers 4 --prefetch-factor 4 \
  --val-pairs 32 --audio-examples 4 \
  --output-dir runs/vc-converter-100k-2gpu-b24-fresh
```

最终文件是 `runs/vc-converter-100k-2gpu-b24-fresh/step-100000.pt`。
两卡 global batch48，Converter 学习率沿用 `1e-4`；保存间隔 5000、验证间隔 1000。
不传 `--resume` 或 `--initialize-vc`。中断后同 batch 续训才使用 `--resume`；
此时 `--steps` 是新增更新数，例如 step30000 后传 `--steps 70000`，到累计 100000。
续训时使用该运行保存的 `config.yaml`，不要重新生成不同的配置。

## 训练吞吐

- 相同有效时长的 reference 合并运行冻结 ERes2Net 和 Mel；不同长度仍分别裁剪，避免把 padding 池化成音色。
  大量 reference 达到 6s 上限时收益更明显；X-VC 的 FBank 内部仍按样本提取。
- VC trainer 不构造、预取或传输未使用的整句 `source_wav_cond/target_wav_cond`；reference 定义不变。
  Converter 阶段也不将未使用的 `natural` crop 搬到 GPU。IO 检查工具保留历史条件字段。
- `num-workers` 按每卡计；4 workers × 2 卡共 8 个进程，prefetch4 为每卡最多 16 个预取 batch。
  保留 persistent workers、pinned memory 和 nonblocking 传输。若数据等待占比高且 CPU/存储允许，可试每卡 8 workers。
- 日志中的 `global_pairs_per_second` 衡量吞吐，`mean_data_wait_seconds_by_rank` 和
  `mean_step_seconds_by_rank` 用于区分输入等待和训练开销；40% 显存占用本身不是 GPU 利用率。
- validation 仍每 1000 步仅在 rank0 上运行，另一卡等待；此时的单卡低 util 属于预期。
  日志吞吐排除显式验证、保存和日志通信时间，因此还需观察整个运行的总耗时。

以上优化不代表已在服务器 GPU 上测得加速比例；用同一设备比较修改前后的吞吐和数据等待。

## 输入与参考

保留原 X-VC JSONL 的五个字段：`index/source_utt/target_utt/source_wav_path/target_wav_path`。
默认仍是 40% generated→natural、20% natural 自重建、40% natural→generated。
可以增加两个字段，推荐用于正式训练和独立参考验证：

```json
{"index":0,"source_utt":"generated_0","target_utt":"natural_0","source_wav_path":"/data/generated.flac","target_wav_path":"/data/natural.wav","target_reference_wav_path":"/data/natural_same_speaker_other_utt.wav","source_reference_wav_path":"/data/generated_voice_prompt_other_utt.wav"}
```

- 正向与 natural 自重建使用 `target_reference_wav_path`。
- 反向使用 `source_reference_wav_path`；必须属于生成 source 的实际声音身份，可从 Seed-VC
  的生成 prompt 元数据取得。不能从 `source_utt` 的数字后缀推断。
- 两个 reference 应与配对音频是不同句子；代码拒绝相同文件路径，但不能从路径证明身份或文本。
- 没有 reference 字段时，使用 loader 分配后的 target 音频，移除训练 crop 后的剩余部分。
  speaker embedding 与 Mel 使用同一个参考；这属于 `paired_masked` 条件，不是独立异句参考。
  `mask_target_condition:false` 则是 `paired_full`，会包含监督片段，只适合明确标注的配对条件基线。
- reference 最长取前 6 秒，必须至少有 0.5 秒有效、非静音音频，推荐 2–6 秒。
  短句或验证整句被 crop 覆盖时，必须提供独立 reference；不会把全零条件静默当作目标音色。

`natural` 始终指原 JSONL 的自然 target crop，正向/反向采样不会改变其身份；第二阶段普通
自重建始终走 `E(natural) → Fusion → Decoder`，不经过 Converter。

train/val 应按原自然 target 分组；同一 target 的所有生成变体留在同一 split。
入口拒绝两份 manifest 中重复的原 `target_utt` 或 `target_wav_path`。
入口不验证 speaker ID。自动从同目录选取 reference 的脚本见
[VC_PAIR_DATA.md](VC_PAIR_DATA.md)：该规则是 target utterance holdout，允许已见 speaker 和
train 中另一句音频作为 reference，不划分独立 reference pool。

配对监督仍假设相同内容且局部时间轴已对齐。loader 只做共同 hop crop，不估计时差、不做 DTW，
Seed-VC 生成关系本身不能证明帧对齐。帧监督使用 Student/Codec 共同有效长度；padding 不参与 loss。

## 目标函数

仅使用三组约束，权重见 YAML：

1. 预测 dyn/edit 对冻结 target latent 的 masked Smooth-L1；inv 始终保持 source，不优化 inv。
2. 配对波形复用现有 `ReconstructionLoss`：log spectral、spectral convergence、log Mel，
   加 `0.1 × waveform L1`。因此该训练需要实际时间对齐，不能用于不同内容 reference 的波形匹配。
3. 仅 Decoder 阶段加入自然音频普通自重建，复用相同 reconstruction loss。

当前没有加入 GAN、speaker waveform loss、cycle 或 GRL。低学习率 Decoder 微调与普通自重建
用于限制漂移，是否提升最终音质仍需听固定样例；不能把训练 loss 下降当作 VC 已成功。
`paired_reconstruction/dyn/edit/natural_reconstruction` 是训练诊断，不是 WER 或 speaker SIM。
试跑默认权重和步数没有经过 6000h 训练验证。

## 环境与命令

复用 X-VC 的 ERes2Net 实现，无需复制其网络代码；服务器必须有 X-VC checkout，以及已下载
的 speaker 模型目录。该目录包含 `configuration.json` 和其中 `model.pretrained_model` 指向的权重。
不会自动下载。依赖仍使用 Codec/Student 的 torch、torchaudio、transformers 环境。

在服务器 Codec checkout 中设置以下变量。`CODEC_PT` 是本次 VC 固定使用的模型，
可先选当前 checkpoint，不必等到 600k；请填写实际 manifest 和 speaker 模型路径：

```bash
export PYTHONPATH="$CODEC/src:$STUDENT/src"
export CODEC_PT="$JOINT_RUN/step-300000.pt"
export STUDENT_PT="$STUDENT/runs/student-12x768-2500h-extension-1epoch-v1/step-053168.pt"
export XVC_ROOT=/path/to/X-VC
export SPEAKER_DIR=/path/to/eres2net_model_dir
export VC_TRAIN=/path/to/vc_train.jsonl
export VC_VAL=/path/to/vc_val.jsonl

python -m xvc2_codec.vc_train \
  --config configs/vc_train.yaml \
  --codec-config configs/codec_63m_quality.yaml \
  --codec-checkpoint "$CODEC_PT" --weights ema \
  --student-checkpoint "$STUDENT_PT" \
  --xvc-root "$XVC_ROOT" --speaker-model-dir "$SPEAKER_DIR" \
  --train-manifest "$VC_TRAIN" --val-manifest "$VC_VAL" \
  --stage converter --steps 2000 --batch-size 4 --num-workers 0 \
  --device cuda:0 --output-dir runs/vc-converter-pilot
```

先听 `validation-002000/` 的固定样例，再开始第二阶段；以下 10k 步是试跑预算，不是最优方案：

```bash
python -m xvc2_codec.vc_train \
  --config configs/vc_train.yaml \
  --codec-config configs/codec_63m_quality.yaml \
  --codec-checkpoint "$CODEC_PT" --weights ema \
  --student-checkpoint "$STUDENT_PT" \
  --xvc-root "$XVC_ROOT" --speaker-model-dir "$SPEAKER_DIR" \
  --train-manifest "$VC_TRAIN" --val-manifest "$VC_VAL" \
  --stage decoder --initialize-vc runs/vc-converter-pilot/step-002000.pt \
  --steps 10000 --batch-size 4 --num-workers 0 \
  --device cuda:0 --output-dir runs/vc-decoder-pilot
```

同阶段续训时用 `--resume runs/.../step-XXXXXX.pt` 替换 `--initialize-vc`，其余参数保持一致。
多 GPU 把 `python -m` 换成 `torchrun --standalone --nproc_per_node=4 -m`；batch size 为每卡值。
`--num-workers` 是每卡的音频读取/预处理进程数；`--prefetch-factor` 是每个 worker 预取的 batch 数。
默认保留 `--num-workers 0`，此时不启用预取；有 worker 时自动启用 `persistent_workers`，
并通过共享 epoch 保持跨 epoch 的采样正确。两卡、每卡 batch 16 可先使用：

```bash
--batch-size 16 --num-workers 4 --prefetch-factor 4
```

这会启动总计 8 个 worker，每卡最多预取 16 个 batch。CUDA 训练仍使用 pinned memory 与
non-blocking host→device 拷贝。worker/prefetch 属于运行参数，可在 `--resume` 时调整，
无需改变 YAML、阶段、学习率、batch size 或 world size；旧 checkpoint 也可以继续加载。
已有 checkpoint 续训使用 `--resume`，不要用 `--initialize-vc` 代替同阶段恢复。

日志新增 `mean_data_wait_seconds_by_rank`、`mean_step_seconds_by_rank` 和 `global_pairs_per_second`，
分别观察两卡平均取 batch 等待、训练迭代耗时及整体 pair 吞吐，用相同 batch/数据比较调整效果。
验证目前只在 rank 0 执行，其他 rank 在 barrier 等待；验证期间单卡空闲不会由增加训练 worker 消除。
代码接入 DDP，但本地验证不包含真实多卡 CUDA 性能。

每次验证输出 source、target、当前 Codec 对 target 的自重建、converted、reference 五个 WAV，
附带参考类型/path JSON 和诊断指标；第二阶段另外报告普通自然自重建 loss。
默认验证前 16 个固定样本，输出前 4 组音频。当前没有自动 ASR/SIM 打分。
VC checkpoint 保存 Converter、Decoder、optimizer/scaler 和固定基座信息；不重复保存 Student/encoders，
因此推理必须保留相应 Codec、Student、speaker 模型文件。VC 权重没有额外 EMA，推理读取保存的模型。

## 分块推理

```bash
python -m xvc2_codec.vc_infer \
  --vc-checkpoint runs/vc-decoder-pilot/step-010000.pt \
  --codec-config configs/codec_63m_quality.yaml --xvc-root "$XVC_ROOT" \
  --source /path/to/source.wav --reference /path/to/target_other_utt.wav \
  --chunk-samples 3200 --device cuda:0 --output runs/vc-demo.wav
```

checkpoint 内记录基座文件路径，需要在当前服务器可访问。
reference 离线编码一次，source 经缓存 Student/声学 encoder/Converter/Fusion/Decoder 逐块处理，
最后 flush 只输出双方共同支持的帧。可以用 317 等不规则 chunk 长度验证一致性。
这是文件推理示例：采样率转换、整句音量归一化、高通在分块前执行一次。
真实在线输入不能逐块重复整句归一化，在线前处理尚需固定增益/有状态滤波实现。

## 验证范围

```bash
PYTHONPATH=src:../large_streaming_student/src \
  python -m pytest tests/test_vc_data.py tests/test_vc_streaming.py tests/test_vc_training.py -q
```

测试覆盖冻结范围和实际参数更新、普通自重建对 Decoder 的独立梯度、teacher 稳定、参考角色切换、
padding 不进入 speaker pooling、静态参考 attention 的因果性与分块等价、checkpoint 两阶段衔接，
以及小参数真实 Student/Codec 的训练入口与断点恢复；还检查常驻 worker 的跨 epoch 采样，
以及从 0 worker checkpoint 切换到带预取的 worker 后恢复的一致性。
这些小模型测试不证明真实 ERes2Net/checkpoint 的 VC 质量，也不替代服务器数据/时间对齐检查。
