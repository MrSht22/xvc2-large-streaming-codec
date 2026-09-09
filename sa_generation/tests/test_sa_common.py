from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from sa_common import SourceItem, load_items, prepare_kaldi_data, prepare_normalized_audio


class SACommonTest(unittest.TestCase):
    def test_load_items_resolves_librispeech_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chapter = root / "train-clean-100" / "103" / "1240"
            chapter.mkdir(parents=True)
            audio = chapter / "103-1240-0006.flac"
            sf.write(audio, np.zeros(1600, dtype=np.float32), 16000)
            (chapter / "103-1240.trans.txt").write_text(
                "103-1240-0006 This is   the transcript.\n", encoding="utf-8"
            )
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                '{"utterance_id":"utt","speaker_id":"103","audio_path":"'
                + str(audio)
                + '"}\n',
                encoding="utf-8",
            )

            items = load_items(manifest, None, None, None, "u")
            normalized = {"utt": audio}
            kaldi = root / "kaldi"
            prepare_kaldi_data(items, normalized, kaldi)

            self.assertEqual(items[0].transcript, "This is the transcript.")
            self.assertEqual(
                (kaldi / "text").read_text(encoding="utf-8"),
                "utt This is the transcript.\n",
            )

    def test_prepare_normalized_audio_reuses_valid_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            sf.write(source, np.zeros(1600, dtype=np.float32), 16000)
            item = SourceItem("utt", "spk", str(source))

            outputs = prepare_normalized_audio([item], root / "run")
            with patch("sa_common.normalize_audio") as normalize:
                repeated = prepare_normalized_audio([item], root / "run")

            normalize.assert_not_called()
            self.assertEqual(outputs, repeated)


if __name__ == "__main__":
    unittest.main()
