# 两阶段流式 latent VC

训练入口：`python -m xvc2_codec.vc_train`，配置：`configs/vc_train.yaml`。
Student、KeepHead、AcousticEncoder、Fusion 在两个阶段都冻结，latent teacher 不随 VC 训练改变。
目标 speaker embedding 和 reference Mel 只进入 Converter，Decoder 不接收目标条件。

| 阶段 | 更新模块 | 默认学习率 | 自然音频普通自重建 |
| --- | --- | --- | --- |
| `converter` | Converter，包括 reference encoder/attention | `1e-4` | Decoder 冻结 |
| `decoder` | Converter + 整个 Decoder | `1e-4` / `2e-5` | 每个 batch 都加入 |

第一阶段拿到短训基线后，第二阶段从该 VC checkpoint 初始化，建立新的 optimizer 和阶段内 step。
`--resume` 才是同阶段续训：恢复 optimizer、AMP scaler、epoch 和 batch cursor。
配置、数据路径、batch size、world size、基座 checkpoint 必须一致。角色/crop 的随机采样按
epoch/item 确定，可重放 cursor；多 GPU 使用 DistributedSampler。不要训练中切换到新的 Codec。

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
`cloudstorfs` 建议先维持 `--num-workers 0`。代码接入 DDP，但本地验证不包含真实多卡 CUDA 性能。

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
以及小参数真实 Student/Codec 的训练入口与断点恢复。
这些小模型测试不证明真实 ERes2Net/checkpoint 的 VC 质量，也不替代服务器数据/时间对齐检查。
