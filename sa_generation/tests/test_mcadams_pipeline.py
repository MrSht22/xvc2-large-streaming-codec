from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from mcadams import coefficient_for_identity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class McAdamsPipelineTest(unittest.TestCase):
    def test_identity_coefficient_is_stable(self) -> None:
        self.assertEqual(
            coefficient_for_identity("speaker-1"),
            coefficient_for_identity("speaker-1"),
        )
        self.assertNotEqual(
            coefficient_for_identity("speaker-1"),
            coefficient_for_identity("speaker-2"),
        )

    def test_cli_generates_reproducible_aligned_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_rate = 24000
            duration = 1.2
            t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
            base = (
                0.25 * np.sin(2 * np.pi * 120 * t)
                + 0.12 * np.sin(2 * np.pi * 780 * t)
                + 0.05 * np.sin(2 * np.pi * 1900 * t)
            ).astype(np.float32)

            rows = []
            for index, speaker in enumerate(("speaker-a", "speaker-a", "speaker-b")):
                path = root / f"input-{index}.wav"
                stereo = np.stack((base, base * 0.8), axis=1)
                sf.write(path, stereo, sample_rate, subtype="PCM_16")
                rows.append(
                    {
                        "utterance_id": f"utt-{index}",
                        "speaker_id": speaker,
                        "audio_path": str(path),
                    }
                )

            manifest = root / "input.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            output_dirs = [root / "run-1", root / "run-2"]
            for output_dir in output_dirs:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(PROJECT_DIR / "anonymize_sa.py"),
                        "--manifest",
                        str(manifest),
                        "--backend",
                        "mcadams",
                        "--output-dir",
                        str(output_dir),
                        "--anonymization-level",
                        "spk",
                    ],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                self.assertIn("anonymization=PASS", result.stdout)

            pairs = [
                json.loads(line)
                for line in (output_dirs[0] / "pairs.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(pairs), 3)
            self.assertEqual(pairs[0]["mcadams_coefficient"], pairs[1]["mcadams_coefficient"])
            self.assertNotEqual(pairs[0]["mcadams_coefficient"], pairs[2]["mcadams_coefficient"])
            for pair in pairs:
                self.assertEqual(pair["source_audio"]["sample_rate"], 16000)
                self.assertEqual(pair["anonymized_audio"]["sample_rate"], 16000)
                self.assertEqual(
                    pair["source_audio"]["num_frames"],
                    pair["anonymized_audio"]["num_frames"],
                )
                utterance = pair["utterance_id"]
                self.assertNotEqual(
                    sha256(output_dirs[0] / "normalized" / f"{utterance}.wav"),
                    sha256(output_dirs[0] / "anonymized" / f"{utterance}.wav"),
                )
                self.assertEqual(
                    sha256(output_dirs[0] / "anonymized" / f"{utterance}.wav"),
                    sha256(output_dirs[1] / "anonymized" / f"{utterance}.wav"),
                )


if __name__ == "__main__":
    unittest.main()
