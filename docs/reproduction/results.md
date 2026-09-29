# Reproduction Results

**Verdict: REPRODUCTION PASS.**

The official GroundingDINO-T COCO val2017 zero-shot evaluation was reproduced using the official
repository, config, pretrained checkpoint and evaluation script, all unmodified.

```text
Reproduced COCO box AP : 48.493
Official README        : 48.5    ->  delta -0.007
Official ckpt table    : 48.4    ->  delta +0.093
```

No training or fine-tuning is involved. This is an evaluation of the officially released
pretrained model.

---

## Full evaluator output

Verbatim from `demo/test_ap_on_coco.py`:

```text
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.644
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.529
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.339
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.518
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.634
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.386
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.669
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.737
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.775
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.887
```

Raw `stats` array as printed by the evaluator:

```text
Final results: [0.48492997796649423, 0.6440952048284158, 0.5294343500040071,
                0.33938899127414984, 0.5178110613315107, 0.6337227559823642,
                0.3859391586851767, 0.6685007493946463, 0.7374432801030781,
                0.5900804431393171, 0.7751257008188694, 0.8873085802020343]
```

Full log: [`results/logs/coco_val2017_evaluation.log`](../../results/logs/coco_val2017_evaluation.log).
Machine-readable: [`results/coco_reproduction.json`](../../results/coco_reproduction.json).

## Comparison

| Metric | Official / reference | Reproduced | Delta |
|---|---|---|---|
| **AP** @[0.50:0.95] | **48.5** (README) / **48.4** (table) | **48.493** | **−0.007** / **+0.093** |
| AP50 | not published separately | 64.410 | — |
| AP75 | not published separately | 52.943 | — |
| APS | not published separately | 33.939 | — |
| APM | not published separately | 51.781 | — |
| APL | not published separately | 63.372 | — |
| AR@100 | not published separately | 73.744 | — |

Only the primary AP has a published counterpart. The remaining figures are recorded as our
reproduced baseline for future comparison, not as matches against anything.

## Why 48.493 counts as a match

The gap to the README's stated result is **0.007 AP** — about one seventieth of the 0.1 AP
spacing between the two figures the upstream project itself publishes (48.4 vs 48.5). It is far
smaller than the inconsistency the upstream documentation already tolerates internally, and
smaller than any plausible effect of the dependency pins.

The value also lands *between* the two published numbers rather than outside them, and closest
to 48.5 — the correct target, because 48.5 is what the README claims for **this specific
evaluation script**, while 48.4 is the model's published zero-shot AP.

No tuning, threshold change, subset selection or rounding was used to reach it. The raw value is
`0.48492997796649423`.

## Run conditions

```text
wall clock   : 29 min 51 s  (start 2026-09-26T14:46:13Z, end 15:16:04Z)
throughput   : ~0.35 s/image, stable
swap used    : 0 B
num_workers  : upstream default, unchanged
images       : all 5000 (no subset)
```

## Attempt history

Three runs were needed. The first two failures were environmental, not methodological, and are
recorded because "it worked on the third try" only means something if the first two are visible.

| Attempt | Outcome | Cause |
|---|---|---|
| 1 | killed mid-run | Host ran low on memory and the run was reaped. No progress survived: the evaluator prints with plain `print()`, which Python block-buffers through a pipe. |
| 2 | stalled at 959/5000 | The WSL guest was capped at 4 GB. The upstream default `--num_workers 4` forks workers that cannot share the COCO index copy-on-write under Python refcounting (~0.8 GB each); parent + workers ≈ 4.6 GB. Swap exhausted, throughput collapsed from 0.36 s/img to 52 s/img. |
| 3 | **completed** | Host memory reclaimed and the WSL guest raised to 8 GB. Ran to completion with zero swap use. |

**`--num_workers` was never lowered.** Raising the available memory was the point: it let the
documented command run exactly as published, rather than altering the benchmark to fit the
machine.

## Environment

| Component | Version |
|---|---|
| OS | WSL2 Ubuntu 22.04.5 LTS (Windows 11 host) |
| Python | 3.10.12 |
| gcc | 11.4.0 |
| CUDA toolkit | 12.1 (V12.1.105) |
| PyTorch | 2.1.2+cu121 |
| torchvision | 0.16.2+cu121 |
| transformers | 4.36.2 |
| numpy | 1.26.4 |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB |
| Driver | 592.82 |

## Deviations from the official public setup

1. **Platform** — WSL2 Ubuntu 22.04 rather than bare-metal Linux; GPU via WSL CUDA passthrough.
2. **`--no-build-isolation`** added to the documented `pip install -e .`.
3. **Version pins** on dependencies upstream leaves unpinned. Forced by five real
   incompatibilities — see [`installation.md`](installation.md).
4. **Commit pinned to `856dde2`** — upstream publishes no checkpoint-to-commit mapping.
5. **Launch mechanics** — the evaluator was detached from the invoking shell and run with
   `PYTHONUNBUFFERED=1`. Affects scheduling and log flushing only.

**No official model source, config, checkpoint, dataset, or evaluation logic was modified.**
Verified at run time: `git status --porcelain --untracked-files=no` and `git diff HEAD` both
empty; config and checkpoint SHA256 matched the recorded values.
