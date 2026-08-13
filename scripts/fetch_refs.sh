#!/usr/bin/env bash
# Fetch the upstream sources this project reads. They are gitignored because they are
# large and re-fetchable; nothing here is modified in place.
#
# Usage: tools/fetch_refs.sh [dest]   (default: refs/)
set -uo pipefail

DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/refs}"
mkdir -p "$DEST"

TT_REPOS=(
  tt-metal tt-mlir tt-forge tt-forge-fe tt-torch tt-xla tt-llk
  tt-inference-server tt-npe tt-smi tt-topology tt-studio tt-installer
)

for repo in "${TT_REPOS[@]}"; do
  if [ -d "$DEST/$repo" ]; then
    printf '%-22s skip (exists)\n' "$repo"
    continue
  fi
  printf '%-22s ' "$repo"
  ok=0
  for branch in main master; do
    url="https://codeload.github.com/tenstorrent/$repo/tar.gz/refs/heads/$branch"
    if curl -sfSL --max-time 900 -o "/tmp/$repo.tgz" "$url" 2>/dev/null; then
      mkdir -p "$DEST/$repo"
      if tar xzf "/tmp/$repo.tgz" -C "$DEST/$repo" --strip-components=1 2>/dev/null; then
        echo "ok ($branch, $(du -sh "$DEST/$repo" | cut -f1))"
        ok=1
      fi
      rm -f "/tmp/$repo.tgz"
      [ "$ok" -eq 1 ] && break
    fi
  done
  [ "$ok" -eq 0 ] && echo "FAILED"
done

# DeepSeek-V4 modeling code. Weights are not fetched: the checkpoint is ~140 GB and
# bring-up runs against the tiny config in tools/tiny_config.py instead.
HF_DEST="$DEST/hf_dsv4"
mkdir -p "$HF_DEST"
HF_BASE="https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main"
for f in config.json; do
  printf '%-22s ' "hf:$f"
  curl -sfSL -o "$HF_DEST/$f" "$HF_BASE/$f" && echo ok || echo FAILED
done

TF_BASE="https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/deepseek_v4"
for f in configuration_deepseek_v4.py modeling_deepseek_v4.py modular_deepseek_v4.py; do
  printf '%-22s ' "hf:$f"
  curl -sfSL -o "$HF_DEST/$f" "$TF_BASE/$f" && echo ok || echo FAILED
done

echo "done -> $DEST"
