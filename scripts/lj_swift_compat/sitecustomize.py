"""ms-swift / torch<2.6 compatibility shim for the lj containers.

ms-swift's ``swift/callbacks/activation_cpu_offload.py`` does::

    from torch.distributed.fsdp import FSDPModule as FSDP2

``FSDPModule`` (the FSDP2 API) was only added in torch 2.6. The lj Apptainer
images currently ship torch 2.4.1+cu121, so this import fails at *import time*
for the whole ms-swift trainer factory — even though FSDP2 is never used for
LoRA / DDP / DeepSpeed-ZeRO SFT (the recipe used by train_vlm_v2_lj.sh).

This module is auto-imported by CPython at interpreter startup whenever its
directory is on ``PYTHONPATH`` (the ``sitecustomize`` hook). It adds a harmless
placeholder ``FSDPModule`` so the import succeeds. If FSDP2 ever becomes the
active strategy the placeholder would be instantiated and fail loudly, which is
the desired behaviour (we are not using it).

Remove this shim once the lj images ship torch >= 2.6.
"""

try:  # never let the shim break a normal interpreter start
    import torch.distributed.fsdp as _fsdp

    if not hasattr(_fsdp, "FSDPModule"):
        class FSDPModule:  # noqa: D401 - placeholder, FSDP2 unused in LoRA/DDP/ZeRO
            """Placeholder for torch>=2.6 FSDP2; unused by ms-swift LoRA SFT."""

        _fsdp.FSDPModule = FSDPModule
except Exception:  # pragma: no cover - torch missing or unexpected layout
    pass
