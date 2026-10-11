# 流式 latent VC 接口

`xvc2_codec.vc` 实现了 `StreamingLatentConverter` 与 `StreamingVC`。
冻结已训练的 Large Student 和 Codec latent encoders，固定 source inv，由共享因果主干预测 dyn/edit 残差。
目标 condition 只进入 Converter，Fusion/Decoder 保持现有 unconditional 结构。
训练入口现在接入冻结 ERes2Net、静态 reference Mel memory 和两阶段 optimizer，见
[VC_TRAINING.md](VC_TRAINING.md)。没有启动正式训练，也没有训练好的 Converter 权重。

## Converter

输入是 `inv [B,T,128]`、`dyn [B,T,32]`、`edit [B,96,T]` 和 `condition [B,192]`，
latent 维度从 CodecConfig 读取，condition 维度可在构造时指定。
默认 256 channels，六个 3-tap 因果卷积 block，dilation 为 `1,2,4,8,16,32`。
LayerNorm 只在通道维计算，全局 condition 用于逐层 scale/shift 调制。
输出头零初始化，初始 dyn/edit 是输入本身；训练后允许两路一起变换，不给 dyn 加硬幅度限制。

Converter 不需要未来帧，默认历史感受野为 126 帧（2.52 秒）加当前帧；这是有限历史，
不会增加 2.52 秒等待时间。每个 stream 保存自己的卷积 cache 和 condition 副本。
`forward` 用于训练/整段推理；`forward_chunk` 使用 cache，不重算已处理历史，也不做 overlap crossfade。

构造时 `reference_mels=80` 启用 reference encoder 和单向 cross-attention；默认 `0` 保留
只有全局 embedding 的基础接口。source hidden 是 query，离线 reference memory 是 K/V，
reference 不接收 source 信息。`prepare_condition(speaker, mel, mel_lengths)` 返回参考条件；
`init_streaming_state` 保存条件和 K/V 的独立副本，后续 chunk 不重算 reference encoder 或 K/V。

## 配对前向

构造方式（`student` / `codec` 是加载好 checkpoint 的现有对象）：

```python
from xvc2_codec.vc import StreamingLatentConverter, StreamingVC

converter = StreamingLatentConverter(codec.config, condition_dim=192)
vc = StreamingVC(student, codec, converter).to(device)
```

配合 [VCPairDataset](VC_PAIR_DATA.md) 的 batch：

```python
vc.set_training_stage("converter")
vc.train()  # Student/Codec 自动保持 eval 和 frozen；只训练 Converter。
paired = vc.forward_pair(batch, target_condition)
prediction = paired["prediction"]
target_latents = paired["target_latents"]
target_waveform = paired["target_waveform"]
valid_frames = paired["frames"]
```

`source` 经过冻结 Student/声学 encoder → Converter → Fusion/Decoder。
`target` 经过冻结 Student/声学 encoder，产生 detached 的三路 latent teacher；其波形是配对真值。
因此反向/自重建角色自动使用 loader 分配后的输入和监督方向。
`prediction.z_inv` 固定为 source inv；第一版只需监督预测 dyn/edit，target inv 可用于诊断。

`target_condition` 必须由目标声音音频提取，batch 维度与 source 一致。
训练入口的 `ReferenceFrontend` 使用 X-VC 的冻结 ERes2Net，并从同一 reference 提取 Mel；
由 `prepare_condition` 接入 Converter。优先使用同身份异句 reference，不把它当配对真值。

`paired.frames` 排除音频 batch padding 和 Student 的不完整帧，所有 latent/Mel 等配对 loss 必须使用它。
原有效音频只有 320 samples 时 Student 没有有效帧；这样的样本应从 loss 中排除。
Codec 参数不产生梯度，但 Fusion/Decoder 的计算图保留，使波形 loss 可以训练 Converter。
以上是 `converter` 阶段；调用 `set_training_stage("decoder")` 后，Decoder 可训练且跟随
train/eval 模式，Student、KeepHead、AcousticEncoder、Fusion 仍冻结并保持 eval。
`forward_pair(..., reconstruct_natural=True)` 用 `batch.natural` 直接编码/解码自然音频，
绕过 Converter，供 Decoder 的普通自重建约束使用。latent teacher 在两个阶段都保持 detached。

## raw-audio 流式推理

先提供目标参考的全局 embedding，再接收 source stream；参考编码只需计算一次。

```python
vc.eval()
state = vc.init_streaming_state(target_condition)  # [1,192]，与模型相同 device/dtype。
for source_chunk in source_chunks:                # 每块 [1,1,N]，16kHz mono。
    converted_chunk, state = vc.forward_chunk(source_chunk, state)
    consume(converted_chunk)                     # Student 缓冲期间可能是 [1,1,0]。

tail, state = vc.forward_chunk(source_chunk[:, :, :0], state, is_final=True)
consume(tail)
```

可以把 `is_final=True` 放在最后一个非空 chunk 上；若输入流完全为空，flush 使用
`torch.empty(1,1,0,device=device)`。新一句调用 `init_streaming_state` 创建新状态，
finalized 状态禁止复用。多个 stream 可以交错处理，但各自必须持有独立 state。
当前 raw-audio 接口每个 state 支持一条流；整段训练支持 batch。

整段对照接口是 `vc(waveform, sample_lengths, target_condition)`。
输入 chunk 可以在任意 sample 边界切开；接口内部把音频缓冲到完整 Codec hop，再执行
声学 encoder，并等待 Student hidden 后按同一帧序号取 edit，接入 Converter/Fusion/Decoder cache。
flush 交给 Student 输出末尾有效 hidden，清空未被双方共同支持的 raw tail 和多余 edit。
不会用补零凭空合成尾帧。

对于现有 RF=400、stride=320 的 Student，有效输出帧数是
`max(0, floor((N-400)/320)+1)`，VC 输出 samples 为有效帧数乘 320；整段和流式遵守同一规则。
若 Student 为 segment=4、right-context=2，首次非最终输出需要至少
`400+(4+2-1)*320=2000` samples（125ms）的输入；实际延迟还包括输入 chunk 粒度和计算时间。
Converter/Fusion/Decoder 本身不增加未来帧等待。这里没有宣称真实 checkpoint 的实时系数已经达标。

该模型接口接收已经准备好的 16kHz mono waveform，不在每个 chunk 上重新采样、归一化或高通。
X-VC 数据 loader 的整句音量归一化是训练前处理；在线输入须采用固定增益或另行验证的流式前处理。
若在线做高通，应保留滤波状态，不能把每个 chunk 当独立音频重新滤波。

## 验证

在可导入 `xvc2_student` / transformers 的环境中运行：

```bash
PYTHONPATH=src:../large_streaming_student/src \
  python -m pytest tests/test_vc_data.py tests/test_vc_streaming.py -q
```

测试使用真实 Student/Codec 类的小参数结构，不需要下载权重或加载服务器数据。Converter 的
输出头在一致性测试中设为非零，防止仅因初始 identity 而通过。检查范围包括：

- 整段和不规则 chunk 的 latent / waveform 一致性，以及未来 latent 不影响前缀输出。
- 条件更换改变 dyn/edit，inv 保持 source 值。
- 空 chunk、短于 Student RF 的输入、末尾 flush、reset、finalized 状态禁止复用。
- 两条流交错时 cache/条件互不污染，音频/edit 缓冲不随历史长度增长。
- 波形梯度到达 Converter，而 Student/Codec 参数保持冻结。
- 配对前向使用 source inv 与 target latent teacher，并正确处理 batch 的不同有效长度。

这些测试验证实现和流式等价性，不能替代训练后的 VC 音质、身份相似度和实际 GPU 延迟评测。
