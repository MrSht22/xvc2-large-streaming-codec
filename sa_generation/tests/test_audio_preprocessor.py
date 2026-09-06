from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_DIR
    / "third_party"
    / "Voice-Privacy-Challenge-2026"
    / "anonymization"
    / "modules"
    / "sttts"
    / "tts"
    / "IMSToucan"
    / "Preprocessing"
    / "AudioPreprocessor.py"
)
SPEC = importlib.util.spec_from_file_location("voiceprivacy_audio_preprocessor", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
AudioPreprocessor = MODULE.AudioPreprocessor


class AudioPreprocessorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = AudioPreprocessor(input_sr=16000, output_sr=16000)

    def test_short_audio_uses_padded_loudness_analysis(self) -> None:
        audio = np.sin(np.linspace(0, 20, 1600, dtype=np.float32))

        normalized = self.preprocessor.normalize_loudness(audio)

        self.assertEqual(len(normalized), len(audio))
        self.assertTrue(np.isfinite(normalized).all())
        self.assertAlmostEqual(float(np.max(np.abs(normalized))), 1.0, places=6)

    def test_normal_audio_keeps_its_length(self) -> None:
        audio = np.sin(np.linspace(0, 200, 16000, dtype=np.float32))

        normalized = self.preprocessor.normalize_loudness(audio)

        self.assertEqual(len(normalized), len(audio))
        self.assertTrue(np.isfinite(normalized).all())

    def test_silent_audio_remains_finite(self) -> None:
        normalized = self.preprocessor.normalize_loudness(
            np.zeros(1600, dtype=np.float32)
        )

        self.assertEqual(len(normalized), 1600)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertTrue(np.equal(normalized, 0).all())


if __name__ == "__main__":
    unittest.main()
