#!/usr/bin/env python3
"""Build HybPSF v1.0.0 C extensions against the active Pixi CFITSIO.

This script does not modify HybPSF source files or use HybPSF's generated
Makefiles. Run it inside the dedicated ``hybpsf`` Pixi environment.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


PSF_SOURCES = (
    "read_funcpsfs_new.c",
    "fits_IO.c",
    "nrutil.c",
    "creat_basisfs.c",
    "estimate_para.c",
    "fit_gaus.c",
    "gasdev.c",
    "gaus_estimate.c",
    "indexx.c",
    "iSPCA_entr.c",
    "itoa.c",
    "k_sigma.c",
    "lubksb.c",
    "ludcmp.c",
    "moffat.c",
    "moffatlets.c",
    "mov_mer_ba.c",
    "PCA.c",
    "method_branches.c",
    "iSPCA.c",
    "SPCA_entr.c",
    "C_tools_lib.c",
    "ran1.c",
    "svdcmp.c",
    "rebasis.c",
    "star_ellip.c",
    "pythag.c",
    "linear_operation.c",
    "EMPCA.c",
    "EMPCA_entr.c",
    "interp_bicubic.c",
    "SPCA.c",
    "npsf.c",
    "coeff_interp.c",
    "size.c",
    "sort.c",
    "creat_star.c",
    "krig.c",
    "fof_source.c",
)

STAR_SOURCES = (
    "staridf.c",
    "fits_IO.c",
    "nrutil.c",
    "wavelet_Trous.c",
    "sort.c",
    "path_option.c",
    "SNRs.c",
    "fof_source.c",
    "read_funcpsfs.c",
    "ran1.c",
    "gasdev.c",
)


def _run(command: list[str], *, cwd: Path) -> None:
    print("+", shlex.join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _compiler() -> list[str]:
    value = os.environ.get("CC", "cc")
    command = shlex.split(value)
    if not command:
        raise RuntimeError("CC is empty")
    return command


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _build_library(
    *,
    source_dir: Path,
    source_names: tuple[str, ...],
    output_name: str,
    prefix: Path,
) -> Path:
    compiler = _compiler()
    include_dir = _require(prefix / "include")
    library_dir = _require(prefix / "lib")
    _require(library_dir / "libcfitsio.so")

    sources = [_require(source_dir / name) for name in source_names]
    objects: list[Path] = []
    for source in sources:
        obj = source.with_suffix(".o")
        obj.unlink(missing_ok=True)
        _run(
            compiler
            + [
                "-fPIC",
                "-O3",
                "-c",
                source.name,
                "-o",
                obj.name,
                f"-I{include_dir}",
                f"-I{source_dir}",
            ],
            cwd=source_dir,
        )
        objects.append(obj)

    output = source_dir / output_name
    output.unlink(missing_ok=True)
    _run(
        compiler
        + [
            "-shared",
            "-o",
            output.name,
            *[obj.name for obj in objects],
            f"-L{library_dir}",
            f"-Wl,-rpath,{library_dir}",
            "-lcfitsio",
            "-lcurl",
            "-lm",
        ],
        cwd=source_dir,
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hybpsf-root",
        type=Path,
        default=Path("external/HybPSF"),
    )
    args = parser.parse_args()

    prefix_value = os.environ.get("CONDA_PREFIX")
    if not prefix_value:
        raise RuntimeError(
            "CONDA_PREFIX is unset; run with `pixi run -e hybpsf -- ...`"
        )

    root = args.hybpsf_root.expanduser().resolve()
    package = _require(root / "HybPSF")
    prefix = Path(prefix_value).resolve()

    psf_library = _build_library(
        source_dir=_require(package / "source"),
        source_names=PSF_SOURCES,
        output_name="libpsffit.so",
        prefix=prefix,
    )
    star_library = _build_library(
        source_dir=_require(package / "staridf"),
        source_names=STAR_SOURCES,
        output_name="star_extrac.so",
        prefix=prefix,
    )

    print("Built:")
    print(psf_library)
    print(star_library)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
