from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from prepare_audio_manifest import build_rows


class PrepareAudioManifestTest(unittest.TestCase):
    def test_libritts_layout_maps_speaker_and_stable_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = (
                root / "19" / "198" / "19_198_000001.wav",
                root / "19" / "199" / "19_199_000002.flac",
                root / "26" / "251" / "26_251_000001.wav",
            )
            for path in reversed(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            rows = build_rows(root, ("wav", ".flac"), 2, None)

            self.assertEqual([row["speaker_id"] for row in rows], ["19", "19", "26"])
            self.assertEqual(
                [row["utterance_id"] for row in rows],
                [
                    "19__198__19_198_000001",
                    "19__199__19_199_000002",
                    "26__251__26_251_000001",
                ],
            )
            self.assertEqual(rows[0]["relative_audio_path"], "19/198/19_198_000001.wav")

    def test_limit_is_applied_after_deterministic_sort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("c.wav", "a.wav", "b.wav"):
                path = root / "speaker" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            rows = build_rows(root, (".wav",), 1, 2)

            self.assertEqual(
                [row["relative_audio_path"] for row in rows],
                ["speaker/a.wav", "speaker/b.wav"],
            )

    def test_sanitization_collision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("a b.wav", "a-b.wav"):
                path = root / "speaker" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            with self.assertRaisesRegex(ValueError, "duplicate utterance_id"):
                build_rows(root, (".wav",), 1, None)


if __name__ == "__main__":
    unittest.main()
