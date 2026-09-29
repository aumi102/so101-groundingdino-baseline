# Environment

The exact machine the reported numbers came from, and what has to match for them to be
comparable.

---

## Hardware

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 **Laptop**, 8 GB VRAM |
| Driver | 592.82 |
| CPU | 20 logical cores (host) |
| Host RAM | 15.7 GB |

The **Laptop** designation matters. It is power- and thermally-limited; a desktop RTX 4060 will
produce different latency. Any runtime figure in this repository is specific to this GPU at the
stated input resolution.

## Software

| Component | Version |
|---|---|
| Host OS | Windows 11 (build 10.0.26200) |
| Execution environment | WSL2, Ubuntu 22.04.5 LTS |
| Python | 3.10.12 |
| gcc | 11.4.0 |
| CUDA toolkit | 12.1 (V12.1.105) |
| PyTorch | 2.1.2+cu121 |
| torchvision | 0.16.2+cu121 |
| transformers | 4.36.2 |
| tokenizers | 0.15.2 |
| numpy | 1.26.4 |
| opencv-python | 4.9.0.80 |
| pycocotools | 2.0.11 |
| timm | 1.0.30 |
| setuptools | 68.2.2 |

Regenerate this table on any machine with:

```bash
python scripts/verify_environment.py
```

## WSL configuration

The evaluator needs roughly 4.6 GB of guest RAM (see `installation.md`). On WSL2 the guest is
capped by `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=8GB
processors=2
```

With a 4 GB cap the run stalls partway and never completes. With 8 GB it finishes in under half
an hour with zero swap use.

## Dataset layout

Neither the dataset nor the checkpoint is committed to this repository. Expected layout:

```text
<coco_root>/
├── val2017/                      5000 .jpg files
└── annotations/
    └── instances_val2017.json    SHA256 e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f

<gdino_root>/                     the upstream clone at commit 856dde2
└── weights/
    └── groundingdino_swint_ogc.pth
        693,997,677 bytes
        SHA256 3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799
```

Sources:

```bash
# dataset
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# checkpoint
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Validation performed before the evaluation ran:

```text
val2017 image count : 5000
annotations         : 36781
categories          : 80
info.year           : 2017
```

## Determinism and comparability

| Component | Deterministic? | Notes |
|---|---|---|
| COCO evaluation | Yes on identical hardware + versions | `batch_size=1`, `shuffle=False`, model in `eval()` |
| Detector forward | fp32; kernel selection may vary across GPUs and driver versions | |
| Runtime latency | **No** — it is a measurement | Report percentiles, and measure on an idle GPU |

**Measure runtime on an idle GPU.** During earlier work on this project, a contended GPU
inflated the same measurement by roughly 6% and more than doubled its variance. Run-to-run
spread with the GPU exclusively available was 1.4%, so any comparison between two
configurations must clear that margin to mean anything.
