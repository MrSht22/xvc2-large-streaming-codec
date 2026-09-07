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

from sa_common import SourceItem, prepare_normalized_audio


class SACommonTest(unittest.TestCase):
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
