from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from parallel_anonymize_sa import (
    merge_pairs,
    partition_by_speaker,
    shard_output_is_complete,
    worker_environment,
    write_jsonl,
)
from resume_parallel_anonymize_sa import (
    create_resume_log_dir,
    incomplete_shard_indexes,
)


class ParallelAnonymizeSATest(unittest.TestCase):
    def test_worker_environment_binds_one_physical_gpu(self) -> None:
        environment = worker_environment("3")

        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(environment["PYTHONUNBUFFERED"], "1")
        self.assertEqual(
            environment["OMP_NUM_THREADS"], os.environ.get("OMP_NUM_THREADS", "1")
        )
        self.assertEqual(
            environment["MKL_NUM_THREADS"], os.environ.get("MKL_NUM_THREADS", "1")
        )

    def test_partition_keeps_speakers_together_and_balances_rows(self) -> None:
        rows = [
            {"utterance_id": f"a-{index}", "speaker_id": "a"} for index in range(4)
        ] + [
            {"utterance_id": f"b-{index}", "speaker_id": "b"} for index in range(3)
        ] + [
            {"utterance_id": "c-0", "speaker_id": "c"},
            {"utterance_id": "d-0", "speaker_id": "d"},
        ]

        shards = partition_by_speaker(rows, 2)

        speaker_shards = {}
        for shard_index, shard in enumerate(shards):
            for row in shard:
                speaker = row["speaker_id"]
                if speaker in speaker_shards:
                    self.assertEqual(speaker_shards[speaker], shard_index)
                else:
                    speaker_shards[speaker] = shard_index
        self.assertLessEqual(abs(len(shards[0]) - len(shards[1])), 1)

    def test_merge_restores_source_order(self) -> None:
        source_rows = [
            {"utterance_id": "u1", "speaker_id": "a"},
            {"utterance_id": "u2", "speaker_id": "b"},
            {"utterance_id": "u3", "speaker_id": "a"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard_dirs = [root / "shard-0", root / "shard-1"]
            for shard_dir in shard_dirs:
                shard_dir.mkdir()
            (shard_dirs[0] / "pairs.jsonl").write_text(
                json.dumps({"utterance_id": "u2", "speaker_id": "b"}) + "\n"
            )
            (shard_dirs[1] / "pairs.jsonl").write_text(
                "\n".join(
                    json.dumps({"utterance_id": value, "speaker_id": "a"})
                    for value in ("u3", "u1")
                )
                + "\n"
            )
            output = root / "pairs.jsonl"

            merge_pairs(source_rows, shard_dirs, output)

            merged = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["utterance_id"] for row in merged], ["u1", "u2", "u3"])

    def test_complete_shard_is_not_selected_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "shard.jsonl"
            output = root / "output"
            output.mkdir()
            rows = [
                {
                    "utterance_id": "u1",
                    "speaker_id": "a",
                    "sample_alignment": {"source_frames": 100, "aligned_frames": 100},
                },
                {
                    "utterance_id": "u2",
                    "speaker_id": "a",
                    "sample_alignment": {"source_frames": 100, "aligned_frames": 100},
                },
            ]
            write_jsonl(rows, manifest)
            write_jsonl(list(reversed(rows)), output / "pairs.jsonl")

            self.assertTrue(shard_output_is_complete(manifest, output))
            self.assertEqual(incomplete_shard_indexes([manifest], [output]), [])

    def test_legacy_shard_without_sample_alignment_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "shard.jsonl"
            output = root / "output"
            output.mkdir()
            rows = [{"utterance_id": "u1", "speaker_id": "a"}]
            write_jsonl(rows, manifest)
            write_jsonl(rows, output / "pairs.jsonl")

            self.assertFalse(shard_output_is_complete(manifest, output))

    def test_partial_shard_is_selected_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "shard.jsonl"
            output = root / "output"
            output.mkdir()
            rows = [
                {"utterance_id": "u1", "speaker_id": "a"},
                {"utterance_id": "u2", "speaker_id": "a"},
            ]
            write_jsonl(rows, manifest)
            write_jsonl(rows[:1], output / "pairs.jsonl")

            self.assertFalse(shard_output_is_complete(manifest, output))
            self.assertEqual(incomplete_shard_indexes([manifest], [output]), [0])

    def test_resume_log_directory_does_not_overwrite_previous_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_root = Path(temporary)
            first = create_resume_log_dir(log_root)
            (first / "shard-000.log").write_text("old log", encoding="utf-8")
            second = create_resume_log_dir(log_root)

            self.assertEqual(first.name, "resume-001")
            self.assertEqual(second.name, "resume-002")
            self.assertEqual(
                (first / "shard-000.log").read_text(encoding="utf-8"), "old log"
            )


if __name__ == "__main__":
    unittest.main()
