from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda-torch", action="store_true")
    args = parser.parse_args()

    import torch

    configured_root = os.environ.get("CUDAToolkit_ROOT") or os.environ.get("CUDA_HOME")
    root = Path(configured_root).resolve() if configured_root else None
    nvcc = shutil.which("nvcc")
    if nvcc is None and root is not None:
        candidate = root / "bin" / "nvcc"
        if candidate.is_file():
            nvcc = str(candidate)

    headers: list[Path] = []
    libraries: list[Path] = []
    if root is not None:
        headers.extend(
            [
                root / "include" / "cuda.h",
                root / "targets" / "x86_64-linux" / "include" / "cuda.h",
            ]
        )
        libraries.extend(
            [
                *root.glob("lib*/libnvrtc.so*"),
                *root.glob("targets/x86_64-linux/lib*/libnvrtc.so*"),
            ]
        )

    header = _first_existing(headers)
    nvrtc = _first_existing(libraries)
    nvcc_version = ""
    if nvcc is not None:
        result = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, check=False
        )
        nvcc_version = (result.stdout + result.stderr).strip()
    release_match = re.search(r"release\s+(\d+\.\d+)", nvcc_version)
    toolkit_cuda = release_match.group(1) if release_match is not None else None

    payload = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "toolkit_cuda": toolkit_cuda,
        "cuda_available": torch.cuda.is_available(),
        "toolkit_root": str(root) if root is not None else None,
        "nvcc": nvcc,
        "cuda_header": str(header) if header is not None else None,
        "nvrtc": str(nvrtc) if nvrtc is not None else None,
        "nvcc_version": nvcc_version,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    problems = []
    if args.require_cuda_torch and torch.version.cuda is None:
        problems.append(
            "the installed upstream PyTorch wheel is CPU-only; a CUDA-enabled "
            "PyTorch build is required to exercise its TorchConfig CUDA path"
        )
    if torch.version.cuda is not None and toolkit_cuda is not None:
        torch_cuda = ".".join(torch.version.cuda.split(".")[:2])
        if torch_cuda != toolkit_cuda:
            problems.append(
                f"PyTorch expects CUDA {torch_cuda}, but nvcc reports CUDA {toolkit_cuda}"
            )
    if root is None:
        problems.append("CUDA_HOME or CUDAToolkit_ROOT is not set")
    if nvcc is None:
        problems.append("nvcc is unavailable")
    if header is None:
        problems.append("cuda.h is unavailable under the configured toolkit root")
    if nvrtc is None:
        problems.append("NVRTC is unavailable under the configured toolkit root")
    if problems:
        raise SystemExit("CUDA toolkit diagnostic failed: " + "; ".join(problems))


if __name__ == "__main__":
    main()
