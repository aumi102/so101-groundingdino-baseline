#!/usr/bin/env python
"""Verify that this machine can reproduce the recorded Grounding DINO results.

Read-only: installs nothing, builds nothing, changes nothing. Run it before
`reproduce_groundingdino_coco.sh` to find problems in seconds rather than half an hour in.

    python scripts/verify_environment.py --gdino-root /path/to/GroundingDINO --coco-root /path/to/coco

Every check prints OK / WARN / FAIL with the observed value, so a mismatch is actionable rather
than just a refusal. Exit code is 0 only if no check FAILed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

EXPECTED_COMMIT = "856dde20aee659246248e20734ef9ba5214f5e44"
EXPECTED_CONFIG_SHA = "172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1"
EXPECTED_CKPT_SHA = "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
EXPECTED_ANNO_SHA = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
EXPECTED_CKPT_BYTES = 693_997_677
EXPECTED_IMAGES = 5000
EXPECTED_ANNOTATIONS = 36781
EXPECTED_CATEGORIES = 80

#: Versions the recorded results were produced with. A mismatch is a WARN, not a FAIL: other
#: versions may well work, but the numbers are then not strictly comparable.
RECORDED = {
    "python": "3.10.12",
    "torch": "2.1.2+cu121",
    "torchvision": "0.16.2+cu121",
    "transformers": "4.36.2",
    "numpy": "1.26.4",
}

_failed = 0
_warned = 0


def ok(label: str, value: object = "") -> None:
    print(f"  OK    {label:34s} {value}")


def warn(label: str, value: object = "") -> None:
    global _warned
    _warned += 1
    print(f"  WARN  {label:34s} {value}")


def fail(label: str, value: object = "") -> None:
    global _failed
    _failed += 1
    print(f"  FAIL  {label:34s} {value}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_versions() -> None:
    print("\n[versions]")
    actual = platform.python_version()
    (ok if actual == RECORDED["python"] else warn)(
        "python", f"{actual} (recorded {RECORDED['python']})")

    try:
        import torch
    except ImportError:
        fail("torch", "not importable")
        return
    (ok if torch.__version__ == RECORDED["torch"] else warn)(
        "torch", f"{torch.__version__} (recorded {RECORDED['torch']})")

    if torch.cuda.is_available():
        ok("cuda available", f"{torch.cuda.get_device_name(0)} | build {torch.version.cuda}")
    else:
        fail("cuda available", "False -- the evaluation needs a GPU")

    for mod, key in (("torchvision", "torchvision"), ("transformers", "transformers"),
                     ("numpy", "numpy")):
        try:
            v = __import__(mod).__version__
        except ImportError:
            fail(mod, "not importable")
            continue
        (ok if v == RECORDED[key] else warn)(mod, f"{v} (recorded {RECORDED[key]})")

    # numpy 2 against a torch built on the 1.x C API is a known, silent breakage.
    try:
        import numpy
        if int(numpy.__version__.split(".")[0]) >= 2 and torch.__version__.startswith("2.1"):
            fail("numpy/torch ABI", "numpy>=2 with torch 2.1.x -- pin numpy<2")
    except Exception:
        pass


def check_upstream(gdino_root: Optional[Path]) -> None:
    print("\n[upstream checkout]")
    if gdino_root is None or not gdino_root.is_dir():
        warn("GroundingDINO", f"not found at {gdino_root} -- skipping checkout checks")
        return

    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=gdino_root,
                              capture_output=True, text=True, check=True).stdout.strip()
        (ok if head == EXPECTED_COMMIT else warn)("commit", head)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=gdino_root, capture_output=True, text=True,
                               check=True).stdout.strip()
        (ok if not dirty else fail)("tracked files unmodified",
                                    "clean" if not dirty else dirty.replace("\n", "; "))
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        warn("git", f"could not query: {exc}")

    cfg = gdino_root / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    if cfg.is_file():
        digest = sha256(cfg)
        (ok if digest == EXPECTED_CONFIG_SHA else fail)("config sha256", digest)
    else:
        fail("config", f"missing at {cfg}")

    ckpt = gdino_root / "weights/groundingdino_swint_ogc.pth"
    if ckpt.is_file():
        size = ckpt.stat().st_size
        (ok if size == EXPECTED_CKPT_BYTES else fail)("checkpoint bytes", size)
        print("        hashing checkpoint (~700 MB, a few seconds)...")
        digest = sha256(ckpt)
        (ok if digest == EXPECTED_CKPT_SHA else fail)("checkpoint sha256", digest)
    else:
        fail("checkpoint", f"missing at {ckpt}")

    # The compiled CUDA extension is what distinguishes the official implementation from a port.
    # torch must be imported first: _C.so links against libc10.
    try:
        import torch  # noqa: F401
        from groundingdino import _C
        if hasattr(_C, "ms_deform_attn_forward"):
            ok("groundingdino._C", Path(_C.__file__).name)
        else:
            fail("groundingdino._C", "loaded but missing ms_deform_attn_forward")
    except ImportError as exc:
        fail("groundingdino._C", f"{exc} -- see docs/reproduction/installation.md")


def check_dataset(coco_root: Optional[Path]) -> None:
    print("\n[dataset]")
    if coco_root is None or not coco_root.is_dir():
        warn("COCO", f"not found at {coco_root} -- skipping dataset checks")
        return

    images = coco_root / "val2017"
    if images.is_dir():
        n = sum(1 for _ in images.glob("*.jpg"))
        (ok if n == EXPECTED_IMAGES else fail)("val2017 images", n)
    else:
        fail("val2017", f"missing at {images}")

    anno = coco_root / "annotations/instances_val2017.json"
    if anno.is_file():
        print("        hashing annotations...")
        digest = sha256(anno)
        (ok if digest == EXPECTED_ANNO_SHA else warn)("annotations sha256", digest)
        data = json.loads(anno.read_text(encoding="utf-8"))
        (ok if len(data["annotations"]) == EXPECTED_ANNOTATIONS else fail)(
            "annotations", len(data["annotations"]))
        (ok if len(data["categories"]) == EXPECTED_CATEGORIES else fail)(
            "categories", len(data["categories"]))
    else:
        fail("annotations", f"missing at {anno}")


def check_memory() -> None:
    """The upstream default --num_workers 4 needs roughly 4.6 GB; less means thrashing."""
    print("\n[memory]")
    try:
        meminfo = Path("/proc/meminfo").read_text()
        total_kb = int(next(l for l in meminfo.splitlines()
                            if l.startswith("MemTotal")).split()[1])
        total_gb = total_kb / 1024 / 1024
        if total_gb >= 6.0:
            ok("total RAM", f"{total_gb:.1f} GB")
        else:
            warn("total RAM", f"{total_gb:.1f} GB -- the evaluator needs ~4.6 GB for its 4 "
                              "dataloader workers; under ~6 GB it may thrash")
    except Exception:
        warn("total RAM", "could not read /proc/meminfo (non-Linux?)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gdino-root", type=Path, default=Path("GroundingDINO"),
                    help="upstream clone (default: ./GroundingDINO)")
    ap.add_argument("--coco-root", type=Path, default=Path("coco"),
                    help="COCO dataset root (default: ./coco)")
    args = ap.parse_args()

    print("Verifying environment against the recorded reproduction baseline.")
    print("WARN = usable but not strictly comparable. FAIL = will not reproduce.")

    check_versions()
    check_upstream(args.gdino_root)
    check_dataset(args.coco_root)
    check_memory()

    print(f"\n{_failed} failed, {_warned} warnings")
    if _failed:
        print("Resolve the FAILs before running the evaluation. "
              "See docs/reproduction/installation.md.")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
