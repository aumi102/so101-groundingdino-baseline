#!/usr/bin/env python
"""Deployment-side benchmark of the official pretrained Grounding DINO.

Measures latency, and optionally runs a prompt and threshold study, on whatever imagery you
point it at. Uses the upstream `groundingdino.util.inference` API unmodified.

    python scripts/benchmark_detector.py \
        --gdino-root /path/to/GroundingDINO \
        --image frame.jpg \
        --study

Scope. This answers the *deployment* question -- how fast is it, and how does it behave on our
imagery -- which is separate from the COCO reproduction in docs/reproduction/. It performs no
training or fine-tuning: the officially released pretrained checkpoint is evaluated as-is.

It reports no accuracy metric. Accuracy needs labelled ground truth, and for this project none
exists yet; see docs/project_baseline.md.

Measurement rules that make the numbers trustworthy:
  * warm up first -- the first inference pays kernel autotuning and allocator growth, and is not
    representative;
  * synchronise CUDA before stopping the clock, or you time the kernel launch, not the work;
  * report percentiles, because a control loop is broken by its p95 frame, not its mean;
  * run on an idle GPU. A contended GPU inflated this same measurement by ~6% and more than
    doubled its variance during development.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_PROMPT = "box . bottle . can . package ."
INDIVIDUAL_PROMPTS = ["box .", "bottle .", "can .", "package ."]
BOX_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]
TEXT_THRESHOLDS = [0.20, 0.25, 0.30]


def percentile(values: List[float], q: float) -> float:
    """Linear-interpolated percentile; implemented here so numpy's version cannot drift."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (q / 100.0) * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (1.0 - (pos - lo)) + ordered[hi] * (pos - lo)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gdino-root", type=Path, required=True, help="upstream GroundingDINO clone")
    ap.add_argument("--image", type=Path, action="append", required=True,
                    help="image to run on; repeat for several")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help=f"text prompt (default, PROVISIONAL: {DEFAULT_PROMPT!r})")
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=5, help="discarded iterations")
    ap.add_argument("--iters", type=int, default=30, help="timed iterations")
    ap.add_argument("--study", action="store_true",
                    help="also run the prompt and threshold studies")
    ap.add_argument("--annotate-dir", type=Path, default=None,
                    help="write annotated images here")
    ap.add_argument("--out", type=Path, default=None, help="write results JSON here")
    args = ap.parse_args()

    sys.path.insert(0, str(args.gdino_root))
    try:
        import torch
        from groundingdino.util.inference import annotate, load_image, load_model, predict
    except ImportError as exc:
        print(f"ERROR: could not import the upstream package from {args.gdino_root}: {exc}")
        print("See docs/reproduction/installation.md.")
        return 1

    config = args.gdino_root / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    ckpt = args.gdino_root / "weights/groundingdino_swint_ogc.pth"
    for p in (config, ckpt):
        if not p.is_file():
            print(f"ERROR: missing {p}")
            return 1

    images = [p for p in args.image if p.is_file()]
    missing = [p for p in args.image if not p.is_file()]
    for p in missing:
        print(f"WARNING: skipping missing image {p}")
    if not images:
        print("ERROR: no readable images")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device. Latency figures will not be comparable.")

    model = load_model(str(config), str(ckpt), device=device)
    loaded = {p.name: load_image(str(p)) for p in images}

    results: Dict[str, Any] = {
        "_note": "Deployment benchmark of the OFFICIAL PRETRAINED model. No training or "
                 "fine-tuning. Reports no accuracy metric -- no labelled ground truth exists.",
        "prompt": args.prompt,
        "prompt_status": "PROVISIONAL" if args.prompt == DEFAULT_PROMPT else "user-supplied",
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "images": [p.name for p in images],
    }

    def run(tensor, caption: str, box_t: float, text_t: float) -> Dict[str, Any]:
        boxes, logits, phrases = predict(model=model, image=tensor, caption=caption,
                                         box_threshold=box_t, text_threshold=text_t,
                                         device=device)
        return {"n": int(boxes.shape[0]), "phrases": list(phrases),
                "confidences": [round(float(v), 4) for v in logits.tolist()]}

    # ---- detections at the chosen setting ------------------------------------------------
    print(f"\n=== detections (prompt={args.prompt!r}, box={args.box_threshold}, "
          f"text={args.text_threshold}) ===")
    detections = {}
    for p in images:
        _, tensor = loaded[p.name]
        detections[p.name] = run(tensor, args.prompt, args.box_threshold, args.text_threshold)
        print(f"  {p.name:34s} n={detections[p.name]['n']:2d}  {detections[p.name]['phrases']}")
    results["detections"] = detections

    # ---- optional studies ----------------------------------------------------------------
    if args.study:
        print("\n=== prompt study ===")
        prompt_study = {}
        for caption in [args.prompt] + INDIVIDUAL_PROMPTS:
            per = {p.name: run(loaded[p.name][1], caption, args.box_threshold,
                               args.text_threshold) for p in images}
            prompt_study[caption] = per
            print(f"  {caption:34s} -> {sum(v['n'] for v in per.values()):3d} detections")
        results["prompt_study"] = prompt_study

        print("\n=== threshold study ===")
        threshold_study = {}
        for text_t in TEXT_THRESHOLDS:
            for box_t in BOX_THRESHOLDS:
                key = f"box={box_t:.2f},text={text_t:.2f}"
                per = {p.name: run(loaded[p.name][1], args.prompt, box_t, text_t)
                       for p in images}
                threshold_study[key] = per
                confs = [c for v in per.values() for c in v["confidences"]]
                mean_conf = round(statistics.fmean(confs), 4) if confs else None
                print(f"  {key:26s} detections={sum(v['n'] for v in per.values()):3d}  "
                      f"mean_conf={mean_conf}")
        results["threshold_study"] = threshold_study

    # ---- annotated output ----------------------------------------------------------------
    if args.annotate_dir:
        import cv2
        args.annotate_dir.mkdir(parents=True, exist_ok=True)
        print("\n=== annotated ===")
        for p in images:
            source, tensor = loaded[p.name]
            boxes, logits, phrases = predict(model=model, image=tensor, caption=args.prompt,
                                             box_threshold=args.box_threshold,
                                             text_threshold=args.text_threshold, device=device)
            out = args.annotate_dir / f"{p.stem}_annotated.jpg"
            cv2.imwrite(str(out), annotate(image_source=source, boxes=boxes,
                                           logits=logits, phrases=phrases))
            print(f"  {out.name}  ({int(boxes.shape[0])} boxes)")

    # ---- latency -------------------------------------------------------------------------
    print(f"\n=== runtime (warmup={args.warmup} discarded, {args.iters} timed) ===")
    bench = images[0]
    source, tensor = loaded[bench.name]
    for _ in range(args.warmup):
        predict(model=model, image=tensor, caption=args.prompt,
                box_threshold=args.box_threshold, text_threshold=args.text_threshold,
                device=device)
    if device == "cuda":
        torch.cuda.synchronize()

    lat: List[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        predict(model=model, image=tensor, caption=args.prompt,
                box_threshold=args.box_threshold, text_threshold=args.text_threshold,
                device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)

    mean = statistics.fmean(lat)
    runtime = {
        "image": bench.name,
        "source_resolution": f"{source.shape[1]}x{source.shape[0]}",
        "model_input": "shortest edge 800, max 1333 (upstream load_image transform)",
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
        "mean_ms": round(mean, 2),
        "p50_ms": round(percentile(lat, 50), 2),
        "p95_ms": round(percentile(lat, 95), 2),
        "p99_ms": round(percentile(lat, 99), 2),
        "min_ms": round(min(lat), 2),
        "max_ms": round(max(lat), 2),
        "fps_mean": round(1000.0 / mean, 2),
        "fps_p95": round(1000.0 / percentile(lat, 95), 2),
        "measured": "detector forward + postprocess only; no decode, no drawing, no tracking",
    }
    results["runtime"] = runtime
    for k in ("source_resolution", "mean_ms", "p50_ms", "p95_ms", "max_ms",
              "fps_mean", "fps_p95"):
        print(f"  {k:20s} {runtime[k]}")

    period_30fps = 1000.0 / 30.0
    print(f"\n  at 30 FPS the camera period is {period_30fps:.1f} ms; "
          f"p95 latency is {runtime['p95_ms']} ms "
          f"({runtime['p95_ms'] / period_30fps:.1f}x) -- "
          f"roughly 1 frame in {round(runtime['p95_ms'] / period_30fps)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
