# Unified Codec Training Contract

日期：2026-09-04。

## Fresh initialization

以下模块在 step 0 全部随机初始化：

- Keep Head；
- Acoustic Encoder；
- Fusion/Prenet；
- Causal Decoder；
- training-only Z_edit style head；
- complex multi-scale STFT discriminator。

Large Student 已在独立仓库训练并冻结。Codec 仓库读取预计算的 768-d hidden cache，避免每个
Codec step 同时保留 90M Student 与 315M Teacher。

## Loss activation

- C0：source 和 pair view self-reconstruction、style、可用的 phone/dyn/prosody anchor；
- C1：在 C0 基础上逐步加入 adversarial 和 feature matching；
- C2：在 pair batch 上逐步加入强 `Z_inv` SmoothL1 和弱 `Z_dyn` trajectory correlation；
- C3：所有已启用项长期联合，EMA 从 GAN 开始点之后更新。

不对完整 `Z_keep` 使用逐元素强相等，不把 `Z_dyn` 视为需要删除的静态泄露。

## Required acceptance before long training

1. 完整配置参数量与当前记录一致；
2. 单卡 2-step 和四卡 DDP 2-step 均可保存、恢复；
3. 32 条 overfit reconstruction 明显下降且可懂；
4. zero/shuffle latent 消融证明三路 latent 被 Decoder 使用；
5. full/chunk/reset/flush 对真实候选 checkpoint 通过；
6. benchmark 后根据 global audio seconds/update 重算 max_steps。

