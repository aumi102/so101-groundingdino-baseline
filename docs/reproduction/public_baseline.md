# Public Baseline — the reproduction target

Every value here was read from the public source at reproduction time, not from memory.

---

## Target

```text
Paper           : Grounding DINO: Marrying DINO with Grounded Pre-Training for
                  Open-Set Object Detection
Paper URL       : https://arxiv.org/abs/2303.05499
Upstream repo   : https://github.com/IDEA-Research/GroundingDINO
Commit          : 856dde20aee659246248e20734ef9ba5214f5e44   (main, 2024-08-12)
Model           : GroundingDINO-T, Swin-T backbone, O365 + GoldG + Cap4M
Checkpoint      : groundingdino_swint_ogc.pth
Checkpoint URL  : https://github.com/IDEA-Research/GroundingDINO/releases/download/
                  v0.1.0-alpha/groundingdino_swint_ogc.pth
Release         : v0.1.0-alpha, 2023-03-21
Size            : 693,997,677 bytes
SHA256          : 3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799
Config          : groundingdino/config/GroundingDINO_SwinT_OGC.py
Config SHA256   : 172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1
Evaluator       : demo/test_ap_on_coco.py
Dataset         : COCO 2017 val (val2017 + annotations/instances_val2017.json)
Expected metric : COCO zero-shot box AP
Published value : 48.5 (README) / 48.4 (checkpoint table)
```

The checkpoint's declared size matches the GitHub release asset API exactly.

## The official command

From the upstream README, section *COCO Zero-shot Evaluations*:

> We provide an example to evaluate Grounding DINO zero-shot performance on COCO.
> The results should be **48.5**.

```bash
CUDA_VISIBLE_DEVICES=0 \
python demo/test_ap_on_coco.py \
 -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
 -p weights/groundingdino_swint_ogc.pth \
 --anno_path /path/to/annotations/instances_val2017.json \
 --image_dir /path/to/val2017
```

From the same README's *Checkpoints* table, row 1:

| name | backbone | Data | box AP on COCO |
|---|---|---|---|
| GroundingDINO-T | Swin-T | O365, GoldG, Cap4M | **48.4 (zero-shot)** / 57.2 (fine-tune) |

## Why the documentation contains both 48.4 and 48.5

They are two different claims:

* **48.4** — the model's **published zero-shot box AP**, carried in the checkpoint table
  alongside the paper's results.
* **48.5** — what the README states the **bundled example evaluation script** should produce.
  That script arrived later: the README's news entry dated `2023/06/17` reads *"We provide an
  example to evaluate Grounding DINO on COCO zero-shot performance"*, months after the March
  2023 checkpoint release.

So the 0.1 AP gap is between *a published model result* and *a convenience script that
approximates the same evaluation* — not two measurements of one procedure. Since this
reproduction runs that script, **48.5 is the operative target**, and 48.4 is a secondary
reference. Both deltas are reported.

A third number appears in the upstream README and must not be confused with either: it
advertises *"COCO zero-shot 52.5 AP"*. That is a different, larger configuration — **not**
GroundingDINO-T with this checkpoint — and is not the target here.

## Commit ambiguity — stated, not hidden

The checkpoint was published on tag **`v0.1.0-alpha` (2023-03-21)**. The COCO evaluation script
was added around **2023-06-17**, and `main` has since advanced to **`856dde2` (2024-08-12)**.

Upstream publishes **no checkpoint-to-commit mapping**, and the release the checkpoint ships
with predates the evaluation script it is meant to be scored by. There is therefore no single
commit at which both the checkpoint and the documented evaluation command exist as released
artifacts.

> This reproduction pins `856dde20aee659246248e20734ef9ba5214f5e44` — the `main` HEAD carrying
> both `demo/test_ap_on_coco.py` and the README instructions and expected value quoted above.
> It is the self-consistent choice, and it is recorded rather than glossed over.

## What this reproduction does not substitute

* **Not** the HuggingFace `transformers` port of Grounding DINO. That is a different
  implementation; using it would make the result evidence about the port, not about the paper.
  The official repository's compiled CUDA kernels (`groundingdino._C`, MultiScale Deformable
  Attention) are what runs here — verified at load time.
* **Not** a custom evaluator, a re-implemented COCO metric, a modified config, or a subset of
  val2017.
