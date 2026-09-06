# VoicePrivacy SA repository and deployment report

Research and availability were checked on 2026-08-08.

## Conclusion

Two complete, citable deployment paths are included.

| Path | Historical identity | Weights | Deployment status |
|---|---|---:|---|
| `mcadams` | VoicePrivacy 2022 official B2 | None | Verified by automated tests |
| `sttts` | VoicePrivacy 2022 submission T04; later VoicePrivacy 2026 B3 | Public v2.0 | Full CPU smoke verified |

The 2022 B1.a/B1.b neural baselines were not selected. Their official source is
public, but `models.2022.tar.gz` is fetched through credentialed SFTP after
organizer registration. A GitHub repository alone is therefore insufficient
for a self-contained deployment package.

## Repository provenance

- VoicePrivacy 2022 official repository:
  https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022
- VoicePrivacy 2026 maintained implementation used here:
  https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2026
- Vendored commit: `f59d6282fddd6c09d5dba5c261a180334ce2797d`
- Original T04 implementation:
  https://github.com/DigitalPhonetics/speaker-anonymization
- VoicePAT toolkit:
  https://github.com/DigitalPhonetics/VoicePAT
- Public neural model release:
  https://github.com/DigitalPhonetics/speaker-anonymization/releases/tag/v2.0

The vendored code is GPL v3 or later. See `third_party/Voice-Privacy-Challenge-2026/LICENSE`.

## Method 1: VoicePrivacy 2022 B2 McAdams

B2 is a signal-processing baseline. It modifies the poles of the linear
prediction model using a McAdams coefficient, then resynthesizes the waveform.
It needs no checkpoint, GPU, ASR, or TTS. The included wrapper deterministically
maps speaker or utterance IDs to a coefficient in the VoicePrivacy-style range
and produces a pair manifest.

The official 2022 overview/results paper lists B1.a, B1.b, and B2 as the three
2022 baseline systems. B3 is not a VoicePrivacy 2022 baseline identifier.

## Method 2: T04 phonetic STTTS/GAN

T04 decomposes anonymization into:

1. Branchformer hybrid CTC/attention phone recognition.
2. GST-style speaker embedding extraction.
3. WGAN generation and selection of an artificial speaker embedding.
4. Phone-level pitch, energy, and duration extraction.
5. FastSpeech2 prosody-cloning synthesis and HiFi-GAN waveform generation.

This is especially relevant to a content/speaker decomposition experiment:
linguistic content is explicitly represented as a phone sequence while the
speaker representation is replaced. The output is resynthesized and therefore
is not sample-aligned with the source.

The wrapper uses the maintained VoicePrivacy 2026 code, but pins its exact
commit and uses the public DigitalPhonetics v2.0 checkpoint bundle associated
with this STTTS family. This should be cited as a reproduction of the T04/STTTS
method using a maintained later implementation, not as a bit-identical run of
the original 2022 submission.

## Required STTTS models

| Extracted path | Role | Bytes | SHA256 |
|---|---|---:|---|
| `anonymization/gan_style-embed/style-embed_wgan.pt` | WGAN | 949,561 | `4ca939962b7ccfd84e04fbfad1ca5f671a7d78bb008573ec849e741c2fcbd274` |
| `asr/asr_branchformer_tts-phn_en.zip` | phone ASR | 438,409,407 | `1b020d56dc52d66530dc56fcaeb33a28ab989a2ee92a4bcc315076bf70a5154a` |
| `tts/Aligner/aligner.pt` | phone/prosody aligner | 220,647,167 | `b3f653d344d395a6f45cd3b7540aa34f21f32fd4cfc7de0af920cb8ae8dc536e` |
| `tts/Embedding/embedding_function.pt` | GST-style embedding | 1,926,482 | `4fd9656ee5665c46225bb8eded3753403fdedd7e25e143f685684290d04f3427` |
| `tts/FastSpeech2_Multi/prosody_cloning.pt` | prosody-cloning TTS | 566,880,902 | `a682824a97ae2ae1551ad550f3ab3c7ea320fdae63411846bbca3b3efc407946` |
| `tts/HiFiGAN_combined/best.pt` | vocoder | 55,465,709 | `56b8c18a272f66c0cbf9104c4251b06c3be81b306c85be968429fbc1634d1690` |

`prepare_models.py` also records archive sizes, CRC validation, and all hashes
in `model_manifest.json`. The generated pool
`work/model_runtime/style-embed_wgan.pt` is a runtime cache of 5,000 WGAN
samples, not another downloadable checkpoint.

## Verification evidence

The verified English smoke used a 5.08-second 16 kHz mono source on macOS ARM,
PyTorch 2.4.1 CPU, NumPy 1.23.5, and ESPnet 202310. It completed all five neural
stages and produced:

```text
backend=sttts
utterances=1
anonymization=PASS
```

The resulting anonymized WAV was non-empty, mono, 16 kHz, and 4.745 seconds.
This proves single-utterance inference and pair generation; it is not a privacy
or utility benchmark. Before using the generated dataset as SA supervision,
evaluate ASV privacy, ASR/phone utility, coverage, and duration drift on the
target corpus.

## Citations

- Tomashenko et al., *The VoicePrivacy 2022 Challenge: Progress and
  Perspectives in Voice Anonymisation*, arXiv:2407.11516, 2024.
  https://arxiv.org/abs/2407.11516
- Meyer et al., *Speaker Anonymization with Phonetic Intermediate
  Representations*, Interspeech 2022.
  https://www.isca-archive.org/interspeech_2022/meyer22b_interspeech.html
- Meyer et al., *Cascade of Phonetic Speech Recognition, Speaker Embeddings
  GAN and Multispeaker Speech Synthesis for the VoicePrivacy 2022 Challenge*,
  VoicePrivacy 2022 system description T04.
  https://www.voiceprivacychallenge.org/vp2022/docs/3___T04.pdf
- Meyer et al., *Prosody Is Not Identity: A Speaker Anonymization Approach
  Using Prosody Cloning*, ICASSP 2023, DOI 10.1109/ICASSP49357.2023.10096607.
  https://doi.org/10.1109/ICASSP49357.2023.10096607
- Patino et al., *Speaker Anonymisation Using the McAdams Coefficient*,
  Interspeech 2021. https://arxiv.org/abs/2011.01130

When publishing results generated by this package, cite the 2022 overview for
the challenge context, the method paper corresponding to the selected backend,
and the pinned implementation/release provenance above.
