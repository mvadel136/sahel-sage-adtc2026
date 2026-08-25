#!/usr/bin/env bash
# Download the SahelSage model weights.
#
# The evaluator runs this script on a fresh clone, so it must be idempotent,
# need no credentials, and put the file exactly where metadata.json's
# _runtime.model_path says it is.
#
# The URL is pinned to an immutable revision, not a branch: a branch could move
# under the auditor between our self-reported measurements and their run, which
# would put the two outside the comparator's tolerance for reasons nobody could
# reconstruct afterwards.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/sahelsage-v11h-Q4_0-flat.gguf"

# Pinned by commit and checksum at release time
HF_REPO="${SAHEL_HF_REPO:-mvadel136/SahelSage-Qwen3-0.6B}"
HF_REVISION="${SAHEL_HF_REVISION:-9e0b8f4706cef5a311318e68b7337be32b7b8aff}"
MODEL_URL="https://huggingface.co/${HF_REPO}/resolve/${HF_REVISION}/sahelsage-v11h-Q4_0-flat.gguf"

# sha256 of the shipped artifact; verified after download.
EXPECTED_SHA256="${SAHEL_MODEL_SHA256:-5873431bf7f39d716dfc447461280a93248f481821f45b81ac6731f3b5db11e8}"

mkdir -p "$MODEL_DIR"

sha_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    else echo ""; fi
}

verify() {
    [ -n "$EXPECTED_SHA256" ] || return 0
    local actual
    actual="$(sha_of "$1")"
    [ -n "$actual" ] || { echo "note: no sha256 tool found; skipping checksum" >&2; return 0; }
    if [ "$actual" != "$EXPECTED_SHA256" ]; then
        echo "ERROR: checksum mismatch for $1" >&2
        echo "  expected $EXPECTED_SHA256" >&2
        echo "  actual   $actual" >&2
        return 1
    fi
    echo "checksum ok"
}

if [ -f "$MODEL_FILE" ]; then
    echo "model already present: $MODEL_FILE"
    if verify "$MODEL_FILE"; then
        exit 0
    fi
    # A corrupt file must not become a dead end: remove it and re-download,
    # instead of failing identically on every future run.
    echo "removing corrupt file and re-downloading" >&2
    rm -f "$MODEL_FILE"
fi

echo "downloading $MODEL_URL"
TMP="$MODEL_FILE.partial"
if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2 -o "$TMP" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$TMP" "$MODEL_URL"
else
    echo "ERROR: need curl or wget" >&2
    exit 1
fi

# a GGUF must start with the magic bytes; an HTML error page must not pass
if [ "$(head -c 4 "$TMP")" != "GGUF" ]; then
    echo "ERROR: downloaded file is not a GGUF (got an error page?)" >&2
    rm -f "$TMP"
    exit 1
fi

mv "$TMP" "$MODEL_FILE"
if ! verify "$MODEL_FILE"; then
    rm -f "$MODEL_FILE"
    echo "ERROR: fresh download failed its checksum, network corruption or a" >&2
    echo "changed upstream file. Re-run this script; if it persists, check" >&2
    echo "https://huggingface.co/${HF_REPO}" >&2
    exit 1
fi
ls -lh "$MODEL_FILE"
