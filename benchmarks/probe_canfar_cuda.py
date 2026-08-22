#!/usr/bin/env python3
"""Minimal CUDA health probe for CANFAR headless GPU sessions.

Step-by-step flushed output with faulthandler enabled: distinguishes
context-init failures, allocator failures (native vs async backend), and
NVML-only failures such as mem_get_info asserting under MIG slices.
"""

from __future__ import annotations

import faulthandler
import os


def main() -> int:
    faulthandler.enable()
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<default>")
    print(f"step0-alloc-conf {conf}", flush=True)
    import torch

    print(f"step1-torch {torch.__version__} cuda={torch.version.cuda}", flush=True)
    try:
        x = torch.ones(8, device="cuda")
        print("step2-first-cuda-alloc ok", flush=True)
    except BaseException as exc:  # noqa: BLE001 - diagnostic must see everything
        print(f"step2-first-cuda-alloc FAILED {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        return 3
    for _ in range(20):
        _ = torch.nn.functional.layer_norm(x, x.shape[-1])
    torch.cuda.synchronize()
    print("step3-layernorm ok", flush=True)
    big = torch.randn(1024, 1024, device="cuda")
    print(f"step4-bigalloc ok sum={float(big.sum()):.3f}", flush=True)
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"step5-meminfo ok free={free_bytes} total={total_bytes}", flush=True)
    except BaseException as exc:  # noqa: BLE001
        print(f"step5-meminfo raised {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    try:
        name = torch.cuda.get_device_name(0)
        print(f"step6-device-name ok {name}", flush=True)
    except BaseException as exc:  # noqa: BLE001
        print(f"step6-device-name raised {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    after = torch.randn(512, 512, device="cuda")
    print(f"step7-alloc-after-nvml ok sum={float(after.sum()):.3f}", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
