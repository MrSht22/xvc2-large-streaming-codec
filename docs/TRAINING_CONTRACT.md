# Unified Codec Training Contract

日期：2026-09-26。

## Initialization paths

原始 v1 run 在 step 0 将以下模块全部随机初始化：

- Keep Head；
- Acoustic Encoder；
- Fusion/Prenet；
- Causal Decoder；
- training-only Z_edit style head；
- complex multi-scale STFT discriminator。

本轮 disentanglement fork 不重新执行 step 0。`--initialize-from` 从旧 step 60k 加载模型、
discriminator 和兼容 EMA，但不加载 optimizer、scaler 或 RNG；新增 Dynamic/Edit phone
adversary 随机初始化。`--resume` 只用于同一新 run 的严格续训。

Large Student 已在独立仓库训练并冻结。Codec 仓库读取预计算的 768-d hidden cache，避免每个
Codec step 同时保留 90M Student 与 315M Teacher。

## Loss activation

- C0：source 和 pair view self-reconstruction、style 和 `Z_inv` phone anchor；
- C1：在 C0 基础上逐步加入 adversarial 和 feature matching；
- C2：在 pair batch 上逐步加入强 `Z_inv` SmoothL1 和弱 `Z_dyn` trajectory correlation；
- C3：从 fork checkpoint 开始，用 10k step ramp 加入 Dynamic F0/VUV/energy/delta 正监督与
  `Z_dyn`/`Z_edit` weak Phone-GRL；
- C4：选定 late joint checkpoint 后，仅解冻 Decoder 做 reconstruction + MS-STFT GAN/FM
  quality refinement。

不对完整 `Z_keep` 使用逐元素强相等，不把 `Z_dyn` 视为需要删除的静态泄露。
完整 loss、阶段门槛与 checkpoint 语义见
[RETRAINING_PLAN.md](RETRAINING_PLAN.md)。

## Required acceptance before long training

1. 完整配置参数量与当前记录一致；
2. 单卡 2-step 和四卡 DDP 2-step 均可保存、恢复；
3. 32 条 overfit reconstruction 明显下降且可懂；
4. zero/shuffle latent 消融证明三路 latent 被 Decoder 使用；
5. full/chunk/reset/flush 对真实候选 checkpoint 通过；
6. benchmark 后根据 global audio seconds/update 重算 max_steps。
