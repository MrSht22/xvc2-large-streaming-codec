from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from typing import Any


REQUIREMENTS = {
    "PyYAML": (6, 0),
    "torch": (2, 4),
    "torchaudio": (2, 4),
}


def version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.match(r"^(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in numbers.group(1).split(".")) if numbers else ()


def check_environment(require_cuda: bool = False) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    failures: list[str] = []
    for distribution, minimum in REQUIREMENTS.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
            failures.append(f"missing:{distribution}")
            continue
        packages[distribution] = version
        if version_tuple(version) < minimum:
            failures.append(f"version:{distribution}={version}")
    cuda = {"available": False, "device_count": 0}
    if not failures:
        try:
            import torch
            import torchaudio

            if version_tuple(torch.__version__)[:2] != version_tuple(torchaudio.__version__)[:2]:
                failures.append("torch_torchaudio_minor_version_mismatch")
            torch.stft(
                torch.zeros(1, 32),
                n_fft=16,
                hop_length=4,
                window=torch.hann_window(16),
                return_complex=True,
            )
            cuda = {
                "available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()),
            }
        except Exception as error:  # pragma: no cover - depends on installed binaries
            failures.append(f"api:{type(error).__name__}:{error}")
    if require_cuda and not cuda["available"]:
        failures.append("cuda_unavailable")
    return {
        "python": sys.version.split()[0],
        "packages": packages,
        "cuda": cuda,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Codec training environment")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = check_environment(args.require_cuda)
    print(json.dumps(report, sort_keys=True))
    print(f"ctc_gop_codec_environment={report['status']}")
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
