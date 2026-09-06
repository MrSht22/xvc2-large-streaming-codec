#!/usr/bin/env python3
"""
Generate libri_{dev,test}_trials_mixed_mcadams from trials_f_mcadams and trials_m_mcadams.
Uses libri_{dev,test}_trials_mixed as reference for utt list and metadata.
"""
import shutil
from pathlib import Path

DATA_DIR = Path("/app/vpc/feb-23/vpc2026-dev/data")
SUFFIX = "_asrbn_hifigan_bn_tdnnf_wav2vec2_vq_48_v1"
PARTITIONS = ["libri_dev", "libri_test"]


def read_kaldi(path):
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            out[parts[0]] = parts[1] if len(parts) > 1 else ""
    return out


def main():
    for base in PARTITIONS:
        ref_mixed = DATA_DIR / f"{base}_trials_mixed"
        out_mixed = DATA_DIR / f"{base}_trials_mixed{SUFFIX}"
        trials_f = DATA_DIR / f"{base}_trials_f{SUFFIX}"
        trials_m = DATA_DIR / f"{base}_trials_m{SUFFIX}"

        if not ref_mixed.exists():
            print(f"Skip {base}: ref {ref_mixed} not found")
            continue
        if not trials_f.exists() or not trials_m.exists():
            print(f"Skip {base}: trials_f or trials_m mcadams not found")
            continue

        # Build utt -> wav path from wav.scp or wav/ dir
        def get_utt2wav(trials_dir):
            wav_scp = read_kaldi(trials_dir / "wav.scp")
            if wav_scp:
                return wav_scp
            wav_dir = trials_dir / "wav"
            if wav_dir.exists():
                return {f.stem: str(f) for f in wav_dir.glob("*.wav")}
            return {}

        f_utts = get_utt2wav(trials_f)
        m_utts = get_utt2wav(trials_m)

        wav_scp_ref = read_kaldi(ref_mixed / "wav.scp")
        utt2spk_ref = read_kaldi(ref_mixed / "utt2spk")
        spk2gender_ref = read_kaldi(ref_mixed / "spk2gender") if (ref_mixed / "spk2gender").exists() else {}
        utt2dur_ref = read_kaldi(ref_mixed / "utt2dur") if (ref_mixed / "utt2dur").exists() else {}
        text_ref = read_kaldi(ref_mixed / "text") if (ref_mixed / "text").exists() else {}

        out_mixed.mkdir(parents=True, exist_ok=True)
        wav_dir = out_mixed / "wav"
        wav_dir.mkdir(exist_ok=True)
        wav_scp_out, utt2spk_out = [], []
        found_utts = set()
        n_from_f, n_from_m, n_miss = 0, 0, 0

        for utt_id in wav_scp_ref:
            src_path = f_utts.get(utt_id) or m_utts.get(utt_id)
            if src_path is None:
                n_miss += 1
                continue
            if utt_id in f_utts:
                n_from_f += 1
            else:
                n_from_m += 1
            found_utts.add(utt_id)
            dst = wav_dir / f"{utt_id}.wav"
            src = Path(src_path)
            if src.exists():
                shutil.copy2(src, dst)
            wav_scp_out.append(f"{utt_id} {dst.resolve()}")
            utt2spk_out.append(f"{utt_id} {utt2spk_ref.get(utt_id, '')}")

        (out_mixed / "wav.scp").write_text("\n".join(wav_scp_out) + "\n")
        (out_mixed / "utt2spk").write_text("\n".join(utt2spk_out) + "\n")
        if spk2gender_ref:
            spks = sorted(set(utt2spk_ref.get(u, "") for u in found_utts if utt2spk_ref.get(u) in spk2gender_ref))
            (out_mixed / "spk2gender").write_text("\n".join(f"{s} {spk2gender_ref[s]}" for s in spks) + "\n")
        if utt2dur_ref:
            (out_mixed / "utt2dur").write_text("\n".join(f"{u} {utt2dur_ref[u]}" for u in found_utts if u in utt2dur_ref) + "\n")
        if text_ref:
            (out_mixed / "text").write_text("\n".join(f"{u} {text_ref[u]}" for u in found_utts if u in text_ref) + "\n")
        if (ref_mixed / "trials").exists():
            trials_out = []
            with open(ref_mixed / "trials") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3 and parts[1] in found_utts:
                        trials_out.append(line.strip())
            if trials_out:
                (out_mixed / "trials").write_text("\n".join(trials_out) + "\n")
        if (ref_mixed / "spk2utt").exists():
            spk2utt_out = {}
            for u in found_utts:
                spk = utt2spk_ref.get(u)
                if spk:
                    spk2utt_out.setdefault(spk, []).append(u)
            (out_mixed / "spk2utt").write_text("\n".join(f"{s} {' '.join(sorted(us))}" for s, us in sorted(spk2utt_out.items())) + "\n")

        print(f"{base}_trials_mixed{SUFFIX}: {len(wav_scp_out)} utts (f={n_from_f}, m={n_from_m}, miss={n_miss})")


if __name__ == "__main__":
    main()
