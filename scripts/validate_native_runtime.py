from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any


ISA_CHOICES = ("auto", "native", "scalar", "avx2", "neon")


def _dispatcher_payload() -> dict[str, Any]:
    import torch

    from torchsky import _cpp, _cpp_ext, _ops

    if not _ops.has_wcs_dispatcher():
        raise AssertionError("WCS dispatcher operators are unavailable")
    if not _ops.has_fixed_dispatcher():
        raise AssertionError("fixed-shape dispatcher operators are unavailable")
    if not _ops.has_spectral_dispatcher():
        raise AssertionError("spectral dispatcher operators are unavailable")

    registrations = {
        "torchsky::wcs_pixel_to_world": ("CPU", "Meta", "Autograd"),
        "torchsky::healpix_ring2nest": ("CPU",),
        "torchsky::spectral_spin_interpolate_recurrence": ("CPU",),
    }
    for qualified, dispatch_keys in registrations.items():
        for dispatch_key in dispatch_keys:
            if not torch._C._dispatch_has_kernel_for_dispatch_key(
                qualified, dispatch_key
            ):
                raise AssertionError(f"{qualified} has no {dispatch_key} kernel")

    pixels = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    converted = torch.ops.torchsky.healpix_ring2nest(1, pixels)

    torch.manual_seed(90210)
    lmax = 8
    nalm = (lmax + 1) * (lmax + 2) // 2
    alms = torch.complex(
        torch.randn(2, nalm, dtype=torch.float64),
        torch.randn(2, nalm, dtype=torch.float64),
    )
    theta = torch.linspace(0.1, torch.pi - 0.1, 15, dtype=torch.float64)
    spectral = torch.ops.torchsky.spectral_spin_interpolate_recurrence(
        alms, theta, lmax, lmax, 2
    )

    first = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float64)
    second = torch.tensor([1.0, -1.5, 4.0], dtype=torch.float64)
    idx = torch.empty((0, 3), dtype=torch.int64)
    coeff = torch.empty(0, dtype=torch.float64)
    world = torch.ops.torchsky.wcs_pixel_to_world(
        first,
        second,
        crpix0=0.0,
        crpix1=0.0,
        cd00=1.0,
        cd01=0.0,
        cd10=0.0,
        cd11=1.0,
        proj_code="CAR",
        lambda_param=1.0,
        hpx_H=4.0,
        hpx_K=3.0,
        east_x=0.0,
        east_y=1.0,
        north_x=0.0,
        north_y=0.0,
        north_z=1.0,
        radial_x=1.0,
        radial_y=0.0,
        radial_z=0.0,
        ra_p_rad=0.0,
        dec_p_rad=0.0,
        phi_p_rad=0.0,
        native_type="center",
        alpha0=0.0,
        delta0=0.0,
        is_tpv=False,
        tpv_idx1=idx,
        tpv_c1=coeff,
        tpv_idx2=idx.clone(),
        tpv_c2=coeff.clone(),
    )
    if not torch.isfinite(spectral).all() or not all(
        torch.isfinite(value).all() for value in world
    ):
        raise AssertionError("native dispatcher invocation produced non-finite values")

    return {
        "isa": _cpp.selected_cpu_isa(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "extension": str(Path(_cpp_ext.__file__).resolve()),
        "fixed": converted.tolist(),
        "spectral_real": spectral.real.reshape(-1).tolist(),
        "spectral_imag": spectral.imag.reshape(-1).tolist(),
        "wcs_first": world[0].tolist(),
        "wcs_second": world[1].tolist(),
    }


def _probe(expected_isa: str | None, emit_json: bool) -> dict[str, Any]:
    payload = _dispatcher_payload()
    if expected_isa is not None and payload["isa"] != expected_isa:
        raise AssertionError(
            f"selected CPU ISA {payload['isa']!r}, expected {expected_isa!r}"
        )
    if emit_json:
        print(json.dumps(payload, separators=(",", ":")))
    else:
        print(
            "native runtime probe passed:",
            f"isa={payload['isa']}",
            f"torch={payload['torch']}",
            f"torch_cuda={payload['torch_cuda']}",
            f"extension={payload['extension']}",
        )
    return payload


def _run_probe(isa: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "probe", "--json"],
        env={**os.environ, "TORCHSKY_CPU_ISA": isa},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if result.returncode in (-signal.SIGILL, 128 + signal.SIGILL):
            raise AssertionError(
                f"TORCHSKY_CPU_ISA={isa} terminated with an illegal instruction"
            )
        raise AssertionError(
            f"TORCHSKY_CPU_ISA={isa} failed with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def _assert_close(first: Any, second: Any, path: str = "payload") -> None:
    if isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            raise AssertionError(
                f"{path} lengths differ: {len(first)} != {len(second)}"
            )
        for index, (left, right) in enumerate(zip(first, second, strict=True)):
            _assert_close(left, right, f"{path}[{index}]")
        return
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        if not math.isclose(float(first), float(second), rel_tol=2e-14, abs_tol=2e-14):
            raise AssertionError(f"{path} differs: {first!r} != {second!r}")
        return
    if first != second:
        raise AssertionError(f"{path} differs: {first!r} != {second!r}")


def _compare(candidate_isa: str, expected_candidate_isa: str | None) -> None:
    scalar = _run_probe("scalar")
    candidate = _run_probe(candidate_isa)
    if scalar["isa"] != "scalar":
        raise AssertionError(f"forced scalar selected {scalar['isa']!r}")
    if (
        expected_candidate_isa is not None
        and candidate["isa"] != expected_candidate_isa
    ):
        raise AssertionError(
            f"forced {candidate_isa} selected {candidate['isa']!r}, "
            f"expected {expected_candidate_isa!r}"
        )
    for key in ("fixed", "spectral_real", "spectral_imag", "wcs_first", "wcs_second"):
        _assert_close(scalar[key], candidate[key], key)
    print(
        "scalar/native parity passed:",
        f"scalar={scalar['isa']}",
        f"candidate={candidate['isa']}",
        f"requested={candidate_isa}",
    )


def _reject(isa: str) -> None:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "probe", "--json"],
        env={**os.environ, "TORCHSKY_CPU_ISA": isa},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        raise AssertionError(
            f"unsupported TORCHSKY_CPU_ISA={isa} unexpectedly succeeded"
        )
    diagnostic = result.stdout + result.stderr
    if "is not supported by this build and CPU/OS" not in diagnostic:
        raise AssertionError(diagnostic)
    print(f"unsupported ISA rejection passed: requested={isa}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe")
    probe.add_argument("--expected-isa", choices=("scalar", "avx2+fma", "neon"))
    probe.add_argument("--json", action="store_true")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--candidate-isa", choices=ISA_CHOICES, required=True)
    compare.add_argument(
        "--expected-candidate-isa", choices=("scalar", "avx2+fma", "neon")
    )

    reject = subparsers.add_parser("reject")
    reject.add_argument("--isa", choices=ISA_CHOICES, required=True)

    args = parser.parse_args()
    if args.command == "probe":
        _probe(args.expected_isa, args.json)
    elif args.command == "compare":
        _compare(args.candidate_isa, args.expected_candidate_isa)
    else:
        _reject(args.isa)


if __name__ == "__main__":
    main()
