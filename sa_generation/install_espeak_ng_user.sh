#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/espeak-ng-1.52.0.tar.gz" >&2
  exit 2
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the target Conda environment before running this script" >&2
  exit 2
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer is intended for the Linux server" >&2
  exit 2
fi

archive="$(realpath "$1")"
expected_sha256="bb4338102ff3b49a81423da8a1a158b420124b055b60fa76cfb4b18677130a23"
actual_sha256="$(sha256sum "$archive" | awk '{print $1}')"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "Unexpected eSpeak-ng archive SHA256: $actual_sha256" >&2
  exit 1
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/espeak-ng-build.XXXXXX")"
cleanup() {
  rm -rf "$build_dir"
}
trap cleanup EXIT

tar -xzf "$archive" -C "$build_dir"
cd "$build_dir/espeak-ng-1.52.0"
touch NEWS AUTHORS ChangeLog
autoreconf --force --install --verbose
./configure \
  --prefix="$CONDA_PREFIX" \
  --libdir="$CONDA_PREFIX/lib" \
  --disable-silent-rules \
  --without-pcaudiolib \
  --without-speechplayer \
  --without-mbrola \
  --without-sonic \
  --without-async

make -j"${ESPEAK_BUILD_JOBS:-4}" src/espeak-ng src/speak-ng
make
make install

library="$(find "$CONDA_PREFIX/lib" -maxdepth 1 -type f -name 'libespeak-ng.so*' | sort | head -1)"
if [[ -z "$library" ]]; then
  echo "eSpeak-ng library was not installed under $CONDA_PREFIX/lib" >&2
  exit 1
fi

activate_dir="$CONDA_PREFIX/etc/conda/activate.d"
deactivate_dir="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$activate_dir" "$deactivate_dir"
printf 'export PHONEMIZER_ESPEAK_LIBRARY="%s"\n' "$library" \
  > "$activate_dir/voiceprivacy_espeak.sh"
printf 'export ESPEAK_DATA_PATH="%s/share/espeak-ng-data"\n' "$CONDA_PREFIX" \
  >> "$activate_dir/voiceprivacy_espeak.sh"
printf 'unset PHONEMIZER_ESPEAK_LIBRARY ESPEAK_DATA_PATH\n' \
  > "$deactivate_dir/voiceprivacy_espeak.sh"

export PHONEMIZER_ESPEAK_LIBRARY="$library"
export ESPEAK_DATA_PATH="$CONDA_PREFIX/share/espeak-ng-data"
espeak-ng --version
echo "PHONEMIZER_ESPEAK_LIBRARY=$PHONEMIZER_ESPEAK_LIBRARY"
echo "ESPEAK_DATA_PATH=$ESPEAK_DATA_PATH"
echo "espeak_user_installation=PASS"
