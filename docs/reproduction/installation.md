# Installation

How the official implementation was built, and the five things that broke on the way.

**The upstream repository is not vendored here.** Clone it yourself, at the pinned commit.

---

## Execution environment

The official implementation compiles a CUDA extension and is Linux-first. On a Windows host a
native build is impractical — a 2023-era CUDA toolkit will reject a current MSVC, and the 2023
C++ sources are not guaranteed against a current PyTorch ATen API. Rather than substitute a
different implementation, provide the platform the code targets:

```bash
wsl --install -d Ubuntu-22.04
```

Ubuntu **22.04** was chosen over newer releases because its Python 3.10 and gcc 11 are
contemporary with the repository and its checkpoint.

```text
Ubuntu 22.04.5 LTS   Python 3.10.12   gcc 11.4.0
CUDA toolkit 12.1 (V12.1.105)
NVIDIA RTX 4060 Laptop, driver 592.82, visible inside WSL
```

CUDA toolkit inside WSL, from NVIDIA's official repository with its GPG keyring:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-1
```

## Installing the official repository

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
git checkout 856dde20aee659246248e20734ef9ba5214f5e44

python -m venv .venv && source .venv/bin/activate
pip install "setuptools==68.2.2" "wheel==0.41.2"
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
pip install -e . --no-build-isolation

# must come AFTER the editable install -- see trap below
pip install "transformers==4.36.2" "opencv-python==4.9.0.80" "numpy<2"

mkdir -p weights && cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

## The five failures, and why they happen

The upstream README's install is three commands. **That sequence does not work in 2026**, because
`requirements.txt` pins nothing —

```text
torch
torchvision
transformers
addict
yapf
timm
numpy
opencv-python
supervision>=0.22.0
pycocotools
```

— so pip resolves every line to a current release, and four of them are incompatible with code
written in 2023. Each failure below was observed, diagnosed, and fixed by constraining the
*environment*. None was fixed by editing official code.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `Failed to initialize NumPy: _ARRAY_API not found` | torch 2.1.2 is built against the NumPy 1.x C API; pip installs NumPy 2.x | `numpy<2` |
| 2 | `ModuleNotFoundError: No module named 'torch'` raised from `setup.py` | `setup.py` does `import torch` at module scope; modern pip builds in an **isolated** env where the venv's torch is invisible | `pip install -e . --no-build-isolation` |
| 3 | `ModuleNotFoundError: No module named 'pkg_resources'` from `torch/utils/cpp_extension.py:28` | setuptools 84 **removed** `pkg_resources`; torch 2.1.2 still imports it | `setuptools==68.2.2` |
| 4 | `ImportError: BertModel requires the PyTorch library but it was not found` | transformers 5.x's `is_torch_available()` rejects torch 2.1.2, so torch-backed classes degrade to stubs | `transformers==4.36.2` |
| 5 | `opencv-python 5.x requires numpy>=2 ... you have numpy 1.26.4` | opencv 5.x needs the NumPy 2 ABI, contradicting #1 | `opencv-python==4.9.0.80` |

### Why torch 2.1.2+cu121 specifically

`requirements.txt` gives no version, so *some* choice was unavoidable. Two concrete reasons
decided it:

1. `demo/test_ap_on_coco.py` calls `torch.load(checkpoint, map_location="cpu")` with no
   `weights_only` argument. **torch 2.6 flipped that default to `True`**, which refuses to
   unpickle this 2023 checkpoint.
2. The `_C` CUDA sources date from 2023 and are not guaranteed to compile against the ATen API
   of current PyTorch.

torch 2.1.2 (Dec 2023) is contemporary with the repository and predates both problems.

`TORCH_CUDA_ARCH_LIST` is deliberately **not** set: `setup.py` already compiles the CUDA sources
because `torch.cuda.is_available()` is true, and letting torch autodetect the architecture is the
untouched default path.

## Verifying the build

This is the check that decides whether the *official* implementation is really running:

```bash
python -c "import torch; from groundingdino import _C; print(_C.__file__, hasattr(_C,'ms_deform_attn_forward'))"
```

Expected:

```text
.../groundingdino/_C.cpython-310-x86_64-linux-gnu.so True
```

The compiled MultiScale Deformable Attention op is present and loads — the official CUDA
kernels, not a fallback and not a port.

> If this fails with `ImportError: libc10.so: cannot open shared object file`, the build is
> probably fine and the *test* is wrong: `_C.so` links against libtorch, so `torch` must be
> imported first. Importing `_C` standalone fails regardless of how correct the build is.

## One trap worth repeating

`pip install -e .` installs `requirements.txt` as a side effect, which **re-installs NumPy 2 over
the pin** — it explicitly uninstalls 1.26.4. The version pins must therefore be applied *after*
the editable install. Applying them first appears to work and silently leaves a broken
environment.

## Memory requirement

The upstream evaluator defaults to `--num_workers 4`. Each DataLoader worker is a forked copy
carrying the loaded COCO annotation index and the torch/CUDA runtime; Python's refcounting
defeats copy-on-write sharing, so each costs roughly **0.8 GB resident**. Parent plus workers
needs about **4.6 GB**.

A machine (or WSL guest) with less than that will thrash and never finish. On WSL this is
governed by `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=8GB
```

Raising the memory is the correct fix. Lowering `--num_workers` would also fit, and cannot change
the resulting AP — `batch_size=1`, `shuffle=False`, model in `eval()`, and workers only prefetch
— but it departs from the documented command, so it was not done here.
