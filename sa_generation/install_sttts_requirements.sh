#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the target Conda environment before running this script" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
requirements="$script_dir/requirements-sttts.txt"
index_url="${STTTS_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

python -m pip install \
  --index-url "$index_url" \
  --timeout 120 \
  "setuptools<81" \
  wheel \
  "Cython==3.2.9" \
  "numpy==1.23.5"

# ESPnet pulls these source distributions. Installing them explicitly without
# build isolation prevents pip from downloading a second build environment.
python -m pip install \
  --index-url "$index_url" \
  --timeout 120 \
  --no-build-isolation \
  "pyworld==0.3.5" \
  "ctc-segmentation==1.7.4"

python -m pip install \
  --index-url "$index_url" \
  --timeout 120 \
  --prefer-binary \
  -r "$requirements"

python -m pip check
echo "sttts_requirements_installation=PASS"
