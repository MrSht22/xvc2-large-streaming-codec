from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import soundfile as sf


PROJECT_DIR = Path(__file__).resolve().parents[1]


class STTTSPipelineTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("SA_STTTS_MODELS_DIR") and os.environ.get("SA_STTTS_TEST_AUDIO"),
        "set SA_STTTS_MODELS_DIR and SA_STTTS_TEST_AUDIO to run the neural smoke test",
    )
    def test_cli_generates_valid_pair(self) -> None:
        models_dir = Path(os.environ["SA_STTTS_MODELS_DIR"]).expanduser().resolve()
        test_audio = Path(os.environ["SA_STTTS_TEST_AUDIO"]).expanduser().resolve()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "sttts-smoke"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_DIR / "anonymize_sa.py"),
                    "--audio",
                    str(test_audio),
                    "--utterance-id",
                    "sttts-smoke",
                    "--speaker-id",
                    "speaker-smoke",
                    "--backend",
                    "sttts",
                    "--models-dir",
                    str(models_dir),
                    "--output-dir",
                    str(output_dir),
                    "--gpus",
                    "cpu",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn("anonymization=PASS", result.stdout)
            rows = [
                json.loads(line)
                for line in (output_dir / "pairs.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 1)
            for key in ("normalized_source_audio_path", "anonymized_audio_path"):
                path = Path(rows[0][key])
                info = sf.info(path)
                self.assertGreater(path.stat().st_size, 44)
                self.assertEqual(info.samplerate, 16000)
                self.assertEqual(info.channels, 1)
                self.assertGreater(info.frames, 0)


if __name__ == "__main__":
    unittest.main()
