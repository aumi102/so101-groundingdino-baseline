# GroundingDINO Baseline for SO-101 Conveyor Grasping

Strict reproduction and perception benchmarking for the capstone
**Visual Servoing-Guided Diffusion Policy for Manipulator Grasping of Quasi-Static Moving
Objects on a Conveyor Belt** (SO-101 arm, Isaac Sim + ROS 2).

---

## Result card

| | |
|---|---|
| Official GroundingDINO COCO zero-shot (README) | **48.5 AP** |
| **Our reproduced result** | **48.493 AP** |
| **Delta** | **−0.007 AP** |
| Dataset | COCO val2017, full 5000 images |
| Hardware | NVIDIA RTX 4060 **Laptop** GPU, driver 592.82 |
| Detector runtime — mean | **243.44 ms** |
| Detector runtime — p95 | **253.58 ms** |
| Detector throughput | **≈ 4.11 FPS** (3.94 FPS at p95) |

> **This repository does not contain a fine-tuned GroundingDINO model.
> The results were obtained by evaluating the official pretrained model.**

No training, fine-tuning, gradient updates or new checkpoints are involved anywhere in this
work. What is reported is a *reproduction* of a published evaluation, plus a *runtime benchmark*
and a *provisional project detector baseline*.

---

## Overview

The capstone needs a perception front-end that can tell a visual-servoing controller where a
graspable object on a conveyor is, and where it will be shortly. Before building anything on top
of a detector, the team's lead asked for the published baseline to be reproduced first:

```text
public baseline  →  reproduce it  →  project detector baseline  →  only then modify
```

This repository covers the first three steps. It deliberately contains no tracker, no motion
estimation, no prediction, no ROS integration and no policy code.

## What has been reproduced

The official **GroundingDINO-T (Swin-T, O365/GoldG/Cap4M)** open-set detector, evaluated
zero-shot on COCO val2017 with the repository's own evaluation script, at a pinned commit, with
the official config and the official released checkpoint.

* Paper: *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object
  Detection* — [arXiv:2303.05499](https://arxiv.org/abs/2303.05499)
* Upstream: [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

## Reproduction results

| Metric | Official / reference | Reproduced |
|---|---|---|
| **AP** @[IoU=0.50:0.95] | **48.5** | **48.493** |
| AP50 | — | 64.410 |
| AP75 | — | 52.943 |
| APS | — | 33.939 |
| APM | — | 51.781 |
| APL | — | 63.372 |

Raw evaluator output for AP: `0.48492997796649423`. Nothing was rounded toward the target.

**On 48.4 vs 48.5.** The upstream documentation carries two numbers, and they are different
claims. **48.4** is the model's published zero-shot box AP in the checkpoint table. **48.5** is
what the README states the *bundled example evaluation script* should produce — that script was
added months after the checkpoint was released. Since this reproduction runs that script, 48.5
is the operative target; the delta against 48.4 is +0.093. A third figure in the upstream README
(52.5 AP) refers to a different, larger configuration and is not the target here.

Full metric table, including all recall figures: [`docs/reproduction/results.md`](docs/reproduction/results.md).
Machine-readable: [`results/coco_reproduction.json`](results/coco_reproduction.json).

## Environment

| Component | Version |
|---|---|
| OS | WSL2 Ubuntu 22.04.5 LTS (on Windows 11) |
| Python | 3.10.12 |
| CUDA toolkit | 12.1 (V12.1.105) |
| PyTorch | 2.1.2+cu121 |
| torchvision | 0.16.2+cu121 |
| transformers | 4.36.2 |
| numpy | 1.26.4 |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB, driver 592.82 |

The dependency versions are pinned deliberately. Upstream `requirements.txt` pins nothing, and
five incompatibilities now exist between the 2023-era code and current releases — see
[`docs/reproduction/installation.md`](docs/reproduction/installation.md), which documents each
failure and its fix.

## Reproduction methodology

```text
official repository  →  pinned commit  →  official config  →  official checkpoint
                     →  COCO val2017   →  official evaluator
```

| Artifact | Identity |
|---|---|
| Repository commit | `856dde20aee659246248e20734ef9ba5214f5e44` |
| Config | `groundingdino/config/GroundingDINO_SwinT_OGC.py` |
| Config SHA256 | `172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1` |
| Checkpoint | `groundingdino_swint_ogc.pth` (release `v0.1.0-alpha`), 693,997,677 bytes |
| Checkpoint SHA256 | `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799` |
| Annotations SHA256 | `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f` |
| Dataset | COCO val2017 — 5000 images, 36781 annotations, 80 categories |

**No official model source, config, checkpoint, dataset, or evaluation logic was modified.**
Verified at run time: `git status --porcelain --untracked-files=no` and `git diff HEAD` were both
empty, and the config and checkpoint hashes matched the values above.

The evaluation ran in **29 min 51 s** over the full 5000 images with the official
`--num_workers` unchanged and zero swap use.

## Runtime benchmark

Measured on an otherwise idle GPU, 5 warm-up iterations discarded, 30 timed iterations, detector
forward + post-process only (no decode, no drawing, no tracking):

| Metric | Value |
|---|---|
| mean | 243.44 ms |
| p50 | 243.88 ms |
| p95 | 253.58 ms |
| p99 | 254.55 ms |
| min / max | 235.05 / 254.69 ms |
| FPS (mean) | 4.11 |
| FPS (p95) | 3.94 |

Input: 1080×1920 source, resized by the official 800/1333 transform.
Machine-readable: [`results/runtime_benchmark.json`](results/runtime_benchmark.json).

### Preliminary implication for visual servoing

```text
30 FPS camera period  =  33.3 ms
GroundingDINO p95     = 253.6 ms
253.6 / 33.3          ≈  7.6
```

**The current GroundingDINO configuration is a detector-side real-time bottleneck for per-frame
30 FPS visual servoing.** It can service roughly one frame in eight.

This is a statement about *this configuration on this hardware*, not a verdict on the approach.
It does not establish that GroundingDINO cannot be used for visual servoing — lower input
resolution, a different backbone, a reduced detection rate, or different hardware have not been
evaluated, and are deliberately out of scope until the baseline is agreed.

## Project detector baseline

Applying the same pretrained model to conveyor-like imagery. **Provisional** settings:

```yaml
prompt:         "box . bottle . can . package ."   # PROVISIONAL
box_threshold:  0.30
text_threshold: 0.25
```

The prompt is provisional because **the team has not yet selected the final conveyor objects**.

Qualitative observations on proxy imagery (public conveyor footage and semi-synthetic
composites — **not** the project's own conveyor):

* `bottle 0.70` — correct detection.
* `box package 0.41` — correct object, but a **compound grounded phrase** spanning two class
  names. Consumers comparing `phrase == "box"` will mis-handle it.
* `can 0.33` — false positive on background machinery.
* A white shampoo bottle in the same frame was **missed**.
* `"can ."` produced no useful detections across the tested proxy images.
* `"package ."` behaves too broadly and fires even on machinery-only imagery.
* Combined prompts can **suppress** detections that appear when classes are queried separately.

These are qualitative baseline observations on proxy data, not statistically validated
conclusions. Detail: [`docs/project_baseline.md`](docs/project_baseline.md).

## Current limitations

* **No project-specific labelled dataset yet.**
* **No project-specific mAP yet.**
* **No fine-tuning has been performed.**
* **No trained project checkpoint exists.**

> Project-specific detection accuracy is not yet measurable because real conveyor footage with
> bounding-box ground truth has not yet been collected and labelled.

The COCO reproduction AP is **not** a project mAP. It measures the published model on a public
benchmark; it says nothing about performance on the capstone's conveyor objects.

## Next step

```text
choose real grasp objects
→ collect conveyor footage
→ annotate held-out dataset
→ finalize vocabulary
→ tune thresholds against real ground truth
→ measure precision / recall / mAP
→ only then begin detector / system modifications
```

## Reproduce

The official repository is **not vendored here**. Clone it separately:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
git checkout 856dde20aee659246248e20734ef9ba5214f5e44

python -m venv .venv && source .venv/bin/activate
pip install "setuptools==68.2.2" "wheel==0.41.2"
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
export CUDA_HOME=/usr/local/cuda-12.1
pip install -e . --no-build-isolation
pip install "transformers==4.36.2" "opencv-python==4.9.0.80" "numpy<2"

mkdir -p weights && cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

COCO val2017:

```bash
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

Expected layout:

```text
<coco_root>/
├── val2017/                      5000 .jpg
└── annotations/
    └── instances_val2017.json
```

Then, from this repository:

```bash
python scripts/verify_environment.py          # checks versions, hashes, dataset, _C extension
GDINO_ROOT=/path/to/GroundingDINO COCO_ROOT=/path/to/coco \
  bash scripts/reproduce_groundingdino_coco.sh
python scripts/benchmark_detector.py --gdino-root /path/to/GroundingDINO --image <image.jpg>
```

`reproduce_groundingdino_coco.sh` is a **wrapper**: it validates the environment and then calls
the upstream evaluator verbatim. It contains no metric logic of its own.

## Repository status

This is the **baseline phase**, before any modification. It exists so that later changes can be
measured against a known, reproduced starting point rather than against an assumption.

Later phases — tracking, motion estimation, prediction, ROS/SO-101 integration, alternative
detectors, and the diffusion policy itself — are **not** in this repository.

## Citation and upstream attribution

GroundingDINO is the work of its authors, not of this project. This repository only reproduces
and benchmarks their released model.

```bibtex
@article{liu2023grounding,
  title   = {Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author  = {Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and
             Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and Zhang, Lei},
  journal = {arXiv preprint arXiv:2303.05499},
  year    = {2023}
}
```

Upstream repository: <https://github.com/IDEA-Research/GroundingDINO> (Apache-2.0).
Original code in *this* repository is MIT-licensed — see [`LICENSE`](LICENSE). No upstream source
is copied here.
