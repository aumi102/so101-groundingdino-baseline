# Detection Checklist — Status

Statuses are exact: `DONE` · `PARTIAL` · `BLOCKED` · `DEFERRED`. Nothing is `DONE` because it
*could* be done — only because evidence exists and is linked.

Two baselines exist and must not be conflated:

* **Official reproduction** — [`reproduction/results.md`](reproduction/results.md), COCO AP **48.493** vs published 48.5.
* **Project deployment baseline** — [`project_baseline.md`](project_baseline.md), qualitative only.

Neither involves training or fine-tuning. Both evaluate the officially released pretrained model.

---

## Summary

| Item | Status |
|---|---|
| requirements / environment | **DONE** |
| GroundingDINO install | **DONE** |
| weights / config | **DONE** |
| GPU / CUDA | **DONE** |
| sample images / video | **PARTIAL** |
| text prompts | **PARTIAL — PROVISIONAL** |
| image inference | **DONE** |
| multiple prompts | **DONE** |
| threshold tuning | **PARTIAL** |
| visualization | **DONE** |
| FPS / latency | **DONE** |
| project accuracy / mAP | **BLOCKED** |
| visual-servoing suitability | **PARTIAL** |

---

## Stage 1 — Setup

### 1. requirements / environment — `DONE`
**Evidence.** WSL2 Ubuntu 22.04.5, Python 3.10.12, gcc 11.4, CUDA 12.1 (V12.1.105), torch
2.1.2+cu121. Five dependency-drift incompatibilities documented with their exact errors.
**Artifact.** [`reproduction/environment.md`](reproduction/environment.md), [`reproduction/installation.md`](reproduction/installation.md), `scripts/verify_environment.py`
**Remaining.** None.

### 2. GroundingDINO install — `DONE`
**Evidence.** Upstream repo at `856dde2` with tracked files unmodified. The compiled CUDA
extension `groundingdino._C` loads with `ms_deform_attn_forward` present — the official kernels,
not a port and not a fallback.
**Artifact.** [`reproduction/installation.md`](reproduction/installation.md)
**Remaining.** None.

### 3. weights / config — `DONE`
**Evidence.** `groundingdino_swint_ogc.pth`, 693,997,677 bytes — an exact match to the GitHub
release asset API — SHA256 `3b3ca256…`. Config SHA256 `172e8001…`.
**Artifact.** [`reproduction/public_baseline.md`](reproduction/public_baseline.md), [`results/coco_reproduction.json`](../results/coco_reproduction.json)
**Remaining.** None.

### 4. GPU / CUDA — `DONE`
**Evidence.** RTX 4060 Laptop 8 GB, driver 592.82, `torch.cuda.is_available() == True` through
WSL CUDA passthrough; full 5000-image evaluation completed on it.
**Artifact.** `scripts/verify_environment.py`, [`results/logs/coco_val2017_evaluation.log`](../results/logs/coco_val2017_evaluation.log)
**Remaining.** None.

### 5. sample images / video — `PARTIAL`
**Evidence.** Public conveyor footage and semi-synthetic composites only.
**Artifact.** [`assets/figures/`](../assets/figures/)
**Remaining.** **No footage of the project's own conveyor exists.** Record the real rig: real
objects, real lighting, real camera placement, across the intended belt-speed range.

### 6. text prompts — `PARTIAL — PROVISIONAL`
**Evidence.** `"box . bottle . can . package ."`. The project's conveyor objects have not been
selected, so this is generic warehouse vocabulary standing in for them.
**Artifact.** [`configs/project_baseline.yaml`](../configs/project_baseline.yaml), [`project_baseline.md`](project_baseline.md) §3
**Remaining.** Choose the actual grasp targets. Measured evidence already argues against the
current set: `can` never fires, and `package` fires on everything including machinery-only
imagery.

## Stage 2 — Project inference baseline

### 7. image inference — `DONE`
**Evidence.** Six conveyor-like images through the official pretrained model; bbox, phrase and
confidence recorded for every one, including the failures.
**Artifact.** [`results/project_prompt_study.json`](../results/project_prompt_study.json), `scripts/benchmark_detector.py`
**Remaining.** Re-run on real project footage when it exists.

### 8. multiple prompts — `DONE`
**Evidence.** Combined prompt vs each individual class. `can .` → 0 detections; `package .` →
fires on all six; the combined prompt returned 0 on one frame where individual prompts returned 3.
**Artifact.** [`results/project_prompt_study.json`](../results/project_prompt_study.json) → `prompt_study`
**Remaining.** Repeat once the real vocabulary is chosen.

### 9. threshold tuning — `PARTIAL`
**Evidence.** `box_threshold` 0.20→0.40 sweeps 23→5 detections, mean confidence 0.334→0.543.
`text_threshold` had no effect at 0.20/0.25/0.30 with single-word class names.
**Artifact.** [`results/project_prompt_study.json`](../results/project_prompt_study.json) → `threshold_study`
**Remaining.** Cannot be *resolved* without labels. The trade-off is mapped; choosing a value
requires ground truth to score against.

### 10. visualization — `DONE`
**Evidence.** Annotated images with box, label and confidence for every input, including the
false positive and the miss.
**Artifact.** [`assets/figures/`](../assets/figures/)
**Remaining.** None.

## Stage 3 — Evaluation

### 11. FPS / latency — `DONE`
**Evidence.** Idle GPU, 5 warm-up discarded, 30 timed: mean 243.44 ms, p50 243.88, p95 253.58,
max 254.69 → 4.11 FPS mean, 3.94 FPS p95.
**Artifact.** [`results/runtime_benchmark.json`](../results/runtime_benchmark.json)
**Remaining.** None for the baseline. Re-measure after any configuration change.

### 12. project accuracy / mAP — `BLOCKED`
**Evidence.** **PROJECT DETECTION ACCURACY NOT YET MEASURABLE** — no labelled project data
exists. No number invented. The COCO reproduction AP is not a project mAP.
**Artifact.** [`project_baseline.md`](project_baseline.md) §7
**Remaining.** Project footage → chosen vocabulary → COCO-format bounding boxes on a held-out
set. Blocked on items 5 and 6.

### 13. visual-servoing suitability — `PARTIAL`
**Evidence.** Quantitative: detector p95 253.6 ms against a 33.3 ms camera period at 30 FPS —
7.6× too slow to run every frame, roughly one frame in eight.
**Artifact.** [`project_baseline.md`](project_baseline.md) §6, [`results/runtime_benchmark.json`](../results/runtime_benchmark.json)
**Remaining.** Needs the controller's real period and the camera specification to close the
latency budget. Establishes a detector-side bottleneck for the per-frame case; does **not**
establish that the approach is unusable.

---

## Explicitly not done in this phase

Tracking, motion estimation, prediction, Kalman filtering, ROS / SO-101 integration, alternative
detectors such as YOLO, resolution sweeps, TensorRT / ONNX / quantisation, Diffusion Policy, and
any model training or fine-tuning.

The agreed sequence is *public baseline → reproduce → project detector baseline → then modify*.
This repository stops at the third step.
