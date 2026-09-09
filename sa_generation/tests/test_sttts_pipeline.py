from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
VENDOR_DIR = PROJECT_DIR / "third_party" / "Voice-Privacy-Challenge-2026"
sys.path.insert(0, str(VENDOR_DIR))

from anonymize_sa import (
    build_sttts_config,
    configure_cuda_visibility,
    enforce_exact_sample_lengths,
)
from anonymization.modules.sttts.text.speech_recognition import SpeechRecognition


class STTTSPipelineTest(unittest.TestCase):
    def test_dataset_transcripts_skip_asr_model_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "utt2spk").write_text("utt speaker\n", encoding="utf-8")
            (data / "text").write_text("utt Ground truth text.\n", encoding="utf-8")

            with patch(
                "anonymization.modules.sttts.text.speech_recognition.create_model_instance"
            ) as create_model:
                recognizer = SpeechRecognition(
                    devices=["cuda:0"],
                    settings={
                        "use_dataset_transcripts": True,
                        "results_path": root / "results",
                    },
                )
                texts = recognizer.recognize_speech(data, "dataset")

            create_model.assert_not_called()
            self.assertFalse(texts.is_phones)
            self.assertEqual(texts["utt"], "Ground truth text.")

    def test_optimized_config_uses_transcripts_without_online_fine_tuning(self) -> None:
        config = build_sttts_config(
            models_dir=Path("models"),
            runtime_vectors=Path("vectors.pt"),
            output_dir=Path("output"),
            dataset_name="sa_input",
            level="spk",
            use_dataset_transcripts=True,
            online_aligner_fine_tune=False,
        )

        self.assertTrue(config["modules"]["asr"]["use_dataset_transcripts"])
        self.assertFalse(config["modules"]["prosody"]["on_line_fine_tune"])

    def test_exact_sample_alignment_only_adjusts_the_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            shorter = root / "shorter.wav"
            longer = root / "longer.wav"
            source_wave = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
            shorter_wave = source_wave[:1500]
            longer_wave = np.pad(source_wave, (0, 100), constant_values=0.25)
            sf.write(source, source_wave, 16000, subtype="FLOAT")
            sf.write(shorter, shorter_wave, 16000, subtype="FLOAT")
            sf.write(longer, longer_wave, 16000, subtype="FLOAT")

            padded = enforce_exact_sample_lengths({"utt": source}, {"utt": shorter})
            trimmed = enforce_exact_sample_lengths({"utt": source}, {"utt": longer})

            padded_wave, _ = sf.read(shorter, dtype="float32")
            trimmed_wave, _ = sf.read(longer, dtype="float32")
            np.testing.assert_array_equal(padded_wave[:1500], shorter_wave)
            np.testing.assert_array_equal(padded_wave[1500:], np.zeros(100, dtype=np.float32))
            np.testing.assert_array_equal(trimmed_wave, source_wave)
            self.assertEqual(padded["utt"]["tail_adjustment_samples"], 100)
            self.assertEqual(trimmed["utt"]["tail_adjustment_samples"], -100)

    def test_exact_sample_alignment_rejects_large_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            anonymized = root / "anonymized.wav"
            sf.write(source, np.zeros(1600, dtype=np.float32), 16000)
            sf.write(anonymized, np.zeros(1200, dtype=np.float32), 16000)

            with self.assertRaisesRegex(RuntimeError, "duration mismatch remains too large"):
                enforce_exact_sample_lengths({"utt": source}, {"utt": anonymized})

    def test_inherited_cuda_visibility_takes_precedence(self) -> None:
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"
            visible = configure_cuda_visibility("0")

            self.assertEqual(visible, ["3"])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous

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
