#!/usr/bin/env bash
# Bootstrap this repo on a QuietBox 2 (Ubuntu 24.04, 4x Blackhole, 256 GB DDR5).
#
# Sets up the host-side analysis and reference tooling only, and verifies it works by
# running the checks that do not need TT hardware. Installing tt-metal/ttnn itself is
# separate and documented upstream:
#   https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md
#
# Usage: scripts/setup_qb2.sh [--skip-refs]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SKIP_REFS=0
[ "${1:-}" = "--skip-refs" ] && SKIP_REFS=1

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

step "Host check"
uname -srm
python3 --version
if command -v tt-smi >/dev/null 2>&1; then
  tt-smi -ls 2>/dev/null | head -20 || echo "tt-smi present but no device listing"
else
  echo "tt-smi not on PATH (fine for host-side work; needed for hardware runs)"
fi
printf 'host RAM: '; awk '/MemTotal/ {printf "%.0f GB\n", $2/1024/1024}' /proc/meminfo
printf 'free disk: '; df -h . | awk 'NR==2 {print $4}'

# The board has one x16-capable slot, so we expect the two p300 cards to negotiate
# different widths (~63 GB/s vs ~8 GB/s). That asymmetry drives expert placement, so
# print it here rather than let anyone assume symmetry. See docs/PLAN.md section 2.
step "PCIe link width per card (drives expert placement)"
if command -v lspci >/dev/null 2>&1; then
  if lspci -d 1e52: >/dev/null 2>&1 && [ -n "$(lspci -d 1e52: 2>/dev/null)" ]; then
    sudo lspci -d 1e52: -vv 2>/dev/null | grep -E '^[0-9a-f]{2}:|LnkCap:|LnkSta:' \
      | sed -E 's/^\s+//' \
      || echo "could not read link status (needs root for LnkSta)"
    echo
    echo "Expect one card at Speed 32GT/s Width x16 and one at 16GT/s x4 or narrower."
    echo "If BOTH are x16, docs/PLAN.md section 2 and section 5 can be simplified -- say so."
  else
    echo "no Tenstorrent devices (vendor 1e52) found; not running on a TT box?"
  fi
else
  echo "lspci not installed: sudo apt install pciutils"
fi

step "Python environment (.venv)"
# A venv works here, unlike the macOS dev box where sandboxing forced a --target install.
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
python - <<'PY'
import numpy, torch, transformers
print(f"torch {torch.__version__} | transformers {transformers.__version__} | numpy {numpy.__version__}")
PY

if [ "$SKIP_REFS" -eq 0 ]; then
  step "Upstream sources (refs/, ~1.4 GB, gitignored)"
  scripts/fetch_refs.sh
else
  step "Upstream sources -- skipped"
fi

step "Golden tensors"
python tools/gen_goldens.py

step "Verify: NumPy reference vs HuggingFace goldens"
python tools/verify_numpy.py

step "Verify: HuggingFace reference vs tt-metal's vendored copy"
# The vendored copy uses relative imports, so its parent must be importable.
if [ -d refs/tt-metal/models/demos/deepseek_v3_d_p/reference ]; then
  PYTHONPATH="tools:refs/tt-metal/models/demos/deepseek_v3_d_p/reference" \
    python tools/xref_ttmetal.py || {
      echo
      echo "If this fails on cache bootstrap, tt-metal's vendored V4 reference has the"
      echo "three defects described in docs/PLAN.md SS6. Apply the fix with:"
      echo "  git -C refs/tt-metal apply $ROOT/patches/tt-metal-v4-cache-bootstrap.patch"
    }
else
  echo "refs/tt-metal not present; skipping (re-run without --skip-refs)"
fi

step "Memory and roofline model"
python tools/budget.py

cat <<EOF

$(printf '\033[1mReady.\033[0m') Activate with:  . .venv/bin/activate

Next, the measurement that gates the performance target (docs/PLAN.md SS9). It needs
host RAM and a model download, but no TT hardware:

  python tools/measure_reuse.py --steps 400
  python tools/expert_cache_sim.py --trace traces/<emitted>.json

EOF
