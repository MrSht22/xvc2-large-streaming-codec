from collections import defaultdict
import os
import shutil

def limit_utts_per_speaker(folder, max_utts_per_spk):
    wav_scp = os.path.join(folder, "wav.scp")
    wav_backup_scp = os.path.join(folder, "wav_backup.scp")

    if not os.path.isfile(wav_scp):
        raise FileNotFoundError(f"wav.scp not found in {folder}")

    # If backup does not exist, create it
    if not os.path.isfile(wav_backup_scp):
        shutil.copy(wav_scp, wav_backup_scp)
        print(f"Created backup: {wav_backup_scp}")
    else:
        print(f"Using existing backup: {wav_backup_scp}")

    spk2utts = defaultdict(list)

    # Read from backup
    with open(wav_backup_scp, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            utt_id, wav_path = line.split(maxsplit=1)
            spk_id = utt_id.split("-")[0]
            spk2utts[spk_id].append((utt_id, wav_path))

    # Write filtered wav.scp
    with open(wav_scp, "w") as f:
        for spk_id, utts in spk2utts.items():
            total_utts = len(utts)

            if total_utts < max_utts_per_spk:
                print(f"Speaker {spk_id}: only {total_utts} utterances, using all")

            for utt_id, wav_path in utts[:max_utts_per_spk]:
                f.write(f"{utt_id} {wav_path}\n")

    print(
        f"Saved filtered wav.scp in {folder} "
        f"(max {max_utts_per_spk} utterances per speaker)"
    )
