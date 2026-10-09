# Seed-VC 同句三路 swap 与 3×3 试听报告

冻结全量 Codec，只生成 `BAA / ABA / AAB`，字母顺序为 inv / dyn / edit。
A 是原音，B 是 Seed-VC 对 A 做的同句 VC。读取已生成的 Seed-VC 音频，
不调用 Seed-VC，不使用 SA，不训练 Codec 或 Converter。

## 配对清单

使用 JSONL，每行一对。路径写绝对路径，`pair_id` 只用字母、数字、`_-.`。

```json
{
  "pair_id": "example_001",
  "donor_method": "seed-vc",
  "text": "The shared English transcript of A and B.",
  "a": {"audio_path": "/data/source.wav"},
  "b": {"audio_path": "/data/seed_vc_output.wav"},
  "a_identity_reference_path": "/data/source_speaker_other_sentence.wav",
  "b_identity_reference_path": "/data/seed_vc_target_reference.wav"
}
```

每个 view 可额外提供 `student_hidden_path`（Tensor 或 `student_hidden` 字典的 `.pt`）；
也支持现有 shard 的 `student_hidden_path / student_hidden_offset_frames /
student_hidden_frames / student_hidden_dim / student_hidden_dtype` 字段。
hidden 必须由该 view 自己的音频提取，不能把 A 的 hidden 复用给 B。
未提供 hidden 时，传 `--student-checkpoint` 在冻结推理中提取两侧 hidden。
本轮固定一份 Codec checkpoint 和一种权重（默认 EMA）。正式结果用 held-out 配对。

身份参考优先使用同身份异句：A 的独立原说话人录音，以及 Seed-VC 使用的自然目标参考。
省略参考时回退到 A/B 原音，报告标记 `same-text fallback`，SIM 有内容混杂。
B 原音的 WER/SIM 是 Seed-VC 质量对照，不被当作完美目标。

## 时间对齐

音频统一为 mono 16 kHz，只移除不足一帧的尾部（<20 ms），不截短整句。
swap 使用 A 帧数，默认将 B 的三路 latent 用同一线性时间映射对齐 A，
不拉伸参考 waveform。保存每对 `donor_frame_map.npy`。

这是归一化时间映射，不是 phone 对齐。优先使用 Seed-VC 保持原时长的设置，
选择局部 phone 边界可靠的同句对。时长相同仍可能局部错位。
若已有更准确对齐，在行中提供 `donor_frame_map_path`：一维 `.npy`，长度为 A 帧数，
值是 B 浮点帧索引，必须单调且在 `[0,T_B-1]` 内。

可选 `alignment_lag_frames` 沿用训练约定：A 起点 `max(-lag,0)`，B 起点 `max(lag,0)`。
非零 lag 必须提供两侧保留内容共同适用的 `aligned_text`；frame map 对应裁剪后帧轴。
不能用 lag 或插值掩盖不同内容/错配 pair。

除三个 swap，生成 A/B 原音、AAA/BBB 原生自重建和 `BBB_mapped`（B 三路均映射到 A）。
BBB 与 BBB_mapped 的 WER/听音差异用于识别映射损伤。

## 运行

生成阶段使用原 `ctc-gop` 环境，不需要 ASR/SV 模型：

```bash
conda activate ctc-gop
cd "$CODEC"
export PYTHONPATH="$CODEC/src:$STUDENT/src"

python -m xvc2_codec.swap_probe generate \
  --manifest /absolute/path/seed_vc_pairs.jsonl \
  --config configs/codec_63m_quality.yaml \
  --checkpoint /absolute/path/codec/step-330000.pt \
  --student-checkpoint /absolute/path/student.pt \
  --weights ema --device cuda:0 \
  --output-dir runs/seed-vc-single-swap/audio-run
```

config 必须匹配 checkpoint 架构，上面路径/step 是模板。已有 hidden 时省略
`--student-checkpoint`；首次可加 `--max-pairs 3`。输出目录必须不存在。
生成清单为 `audio-run/branches.jsonl`，不把这些 pair 加入训练数据。

评测阶段用已有 `codec-eval` 环境，安装 Codec 和 Praat 依赖：

```bash
conda activate codec-eval
cd "$CODEC"
python -m pip install -e '.[prosody]'

python -m xvc2_codec.swap_probe evaluate \
  --manifest runs/seed-vc-single-swap/audio-run/branches.jsonl \
  --evaluator-dir "$ROOT/codec-eval" \
  --models-dir "$ROOT/codec-eval/models" \
  --device cuda --compute-type float16 \
  --examples-per-cell 3 \
  --output-dir runs/seed-vc-single-swap/report
```

复用当前 codec-eval 的 Faster-Whisper large-v3 和 WavLM base-plus SV，不下载模型。
英文 transcript 采用原 codec-eval 的 WER normalization。

## 3×3 的含义

行是三路单独替换；列是输出属性，所有差值相对同一对 AAA。

| 列 | 格子数值（逐条差值均值） | 解读 |
|---|---|---|
| Content | `100 × (WER_swap − WER_AAA)` | 正值为内容破坏，单位 percentage points；同句无法判断内容来自哪路 |
| Local prosody | `RMSE_F0(AAA, BBB_mapped) − RMSE_F0(swap, BBB_mapped)` | 正值为 normalized F0 更接近 B；另列 V/UV、energy、delta 的距离改善 |
| Speaker | `(SIM_B − SIM_A)_swap − (SIM_B − SIM_A)_AAA` | 正值表示朝 B 参考身份移动 |

Prosody 从生成波形用 Praat 提取，沿用训练的 50 Hz 四维定义：逐句 normalized log-F0、
V/UV、normalized log-RMS energy、delta F0。F0 只比较共同 voiced 帧，至少三帧。
AAA/BBB_mapped 的 normalized F0 RMSE <0.1 时该对 prosody 主格记 N/A，子指标保留。
0.1 是预先设定的诊断筛选阈值，不是成功门槛。这里不能证明绝对 pitch register、
总时长或完整节奏已经解耦。

`report.html` 每格默认 3 个样本，覆盖分数分布低、中、高位置，不足时全部显示。
展开播放 A/B、AAA、BBB_mapped 和该 swap，并查看 ASR、SIM、VUV/energy/delta、
RMS ratio 和幅度≥1 比例。音频复制到相邻 `audio/`，整个 report 文件夹可拷贝试听。
颜色只在同列比较，正负方向看列定义；静音/崩坏输出仍保留。

机器可读输出：`metrics.jsonl`、`interventions.jsonl`、`matrix.json`、`matrix.csv`；
生成/评测分别保存 request 配置。混合输出无精确 waveform 真值，不算 PESQ/STOI。
BBB_mapped 是映射控制，不是自然波形 ground truth。
