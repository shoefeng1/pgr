"""Single place that decides which torch device the whole pipeline runs on.

The SAC agent and the diffusion trainer (via `accelerate`) must agree on a device,
otherwise tensors end up split across e.g. `cpu` and `mps` on Apple Silicon.

Selection order:
  1. `set_device(...)` (called by `--device` on the command line), or
  2. the `SYNTHER_DEVICE` environment variable, or
  3. "auto": CUDA if available, otherwise CPU.

"auto" deliberately never picks Apple's `mps` backend: for this repo's small MLPs
(batch size 256) it is often no faster than the CPU, and some ops may be missing.
Opt in explicitly with `--device mps` (and `PYTORCH_ENABLE_MPS_FALLBACK=1` if you hit
"not implemented for MPS" errors).
"""
import os
from typing import Optional

import torch

ENV_VAR = "SYNTHER_DEVICE"


def set_device(preference: Optional[str]) -> None:
    """Record a device preference ("auto", "cpu", "cuda", "cuda:1", "mps")."""
    if preference:
        os.environ[ENV_VAR] = preference


def get_device() -> torch.device:
    preference = os.environ.get(ENV_VAR, "auto").strip().lower()
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{ENV_VAR}={preference!r} but CUDA is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"{ENV_VAR}={preference!r} but the MPS backend is not available.")
    return device
