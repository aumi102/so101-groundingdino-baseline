#!/usr/bin/env bash
#
# Reproduce the official Grounding DINO COCO val2017 zero-shot evaluation.
#
# This is a WRAPPER, not a re-implementation. It validates the environment, resolves paths and
# logs provenance, then invokes `demo/test_ap_on_coco.py` from the upstream repository exactly
# as its README documents. It contains no evaluation logic, no post-processing and no metric
# computation of its own -- if it did, the number it printed would be evidence about this
# script rather than about the published model.
#
# Run it inside a Linux environment (WSL2 works): the upstream implementation compiles a CUDA
# extension and is Linux-first.
#
#   GDINO_ROOT=/path/to/GroundingDINO COCO_ROOT=/path/to/coco \
#     bash scripts/reproduce_groundingdino_coco.sh
#
# Environment variables (all optional if the defaults match your layout):
#   GDINO_ROOT  upstream clone           (default: ./GroundingDINO)
#   COCO_ROOT   dataset root             (default: ./coco)
#   VENV        virtualenv to activate   (default: $GDINO_ROOT/.venv, skipped if absent)
#   CUDA_HOME   CUDA toolkit             (default: /usr/local/cuda)
#   LOG         output log path          (default: ./coco_zero_shot.log)
#
# Expected result: 48.5 AP (upstream README) / 48.4 AP (upstream checkpoint table).
# See docs/reproduction/public_baseline.md for why both numbers exist.

set -euo pipefail

GDINO_ROOT="${GDINO_ROOT:-$PWD/GroundingDINO}"
COCO_ROOT="${COCO_ROOT:-$PWD/coco}"
VENV="${VENV:-$GDINO_ROOT/.venv}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
LOG="${LOG:-$PWD/coco_zero_shot.log}"

# The commit this reproduction was recorded against.
EXPECTED_COMMIT=856dde20aee659246248e20734ef9ba5214f5e44

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- validation --
[ -d "${GDINO_ROOT}" ] || die "upstream checkout not found at ${GDINO_ROOT}
  git clone https://github.com/IDEA-Research/GroundingDINO.git"
[ -d "${COCO_ROOT}/val2017" ] || die "COCO images not found at ${COCO_ROOT}/val2017"
[ -f "${COCO_ROOT}/annotations/instances_val2017.json" ] \
  || die "COCO annotations not found at ${COCO_ROOT}/annotations/instances_val2017.json"
[ -f "${GDINO_ROOT}/weights/groundingdino_swint_ogc.pth" ] \
  || die "checkpoint missing. Fetch it with:
  wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

cd "${GDINO_ROOT}"
if [ -f "${VENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]; then
  echo "WARNING: checkout is at ${ACTUAL_COMMIT}, but this reproduction was recorded against" >&2
  echo "         ${EXPECTED_COMMIT}. The result is not comparable without re-verifying." >&2
fi

# Only TRACKED files matter. Following the upstream README necessarily leaves untracked files:
# `pip install -e .` generates groundingdino/version.py, and the install step creates weights/.
# Blocking on those would refuse to run a correctly-installed upstream repo.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  die "the upstream checkout has modifications to tracked files. A reproduction must run
unmodified code:
$(git status --porcelain --untracked-files=no)"
fi

# The compiled CUDA extension is the difference between running the official implementation and
# silently running something else. torch must be imported first, since _C links against libc10.
python - <<'PY' || die "upstream CUDA extension _C is not importable; see docs/reproduction/installation.md"
import torch
from groundingdino import _C
assert hasattr(_C, "ms_deform_attn_forward"), "_C present but missing ms_deform_attn_forward"
PY

IMAGE_COUNT="$(find "${COCO_ROOT}/val2017" -name '*.jpg' | wc -l)"
[ "${IMAGE_COUNT}" -eq 5000 ] || die "expected 5000 val2017 images, found ${IMAGE_COUNT}.
The published metric is defined on the full split; a subset is a smoke test, not a reproduction."

# ------------------------------------------------------------------ provenance --
{
  echo "=============================================================="
  echo "Grounding DINO official COCO zero-shot evaluation"
  echo "=============================================================="
  echo "start (UTC)  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit       : ${ACTUAL_COMMIT}"
  echo "config sha256: $(sha256sum groundingdino/config/GroundingDINO_SwinT_OGC.py | cut -d' ' -f1)"
  echo "ckpt sha256  : $(sha256sum weights/groundingdino_swint_ogc.pth | cut -d' ' -f1)"
  echo "anno sha256  : $(sha256sum "${COCO_ROOT}/annotations/instances_val2017.json" | cut -d' ' -f1)"
  echo "images       : ${IMAGE_COUNT}"
  echo "gpu          : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
  echo "python       : $(python --version 2>&1)"
  echo "torch        : $(python -c 'import torch;print(torch.__version__, torch.version.cuda)')"
  echo "transformers : $(python -c 'import transformers;print(transformers.__version__)')"
  echo "numpy        : $(python -c 'import numpy;print(numpy.__version__)')"
  echo
  echo "Expected: 48.5 (upstream README) / 48.4 (upstream checkpoint table)."
  echo
} | tee "${LOG}"

# -------------------------------------------------------- the official command --
# Verbatim from the upstream README, "COCO Zero-shot Evaluations".
CUDA_VISIBLE_DEVICES=0 python demo/test_ap_on_coco.py \
 -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
 -p weights/groundingdino_swint_ogc.pth \
 --anno_path "${COCO_ROOT}/annotations/instances_val2017.json" \
 --image_dir "${COCO_ROOT}/val2017" 2>&1 | tee -a "${LOG}"

{
  echo
  echo "end (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "The reported COCO box AP is stats[0] of the 'Final results:' line, i.e."
  echo "Average Precision @[ IoU=0.50:0.95 | area=all | maxDets=100 ]."
} | tee -a "${LOG}"

echo "log written to ${LOG}"
