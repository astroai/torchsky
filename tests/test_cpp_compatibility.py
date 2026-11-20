from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys

import torchsky._cpp as cpp


PACKAGE_ROOT = Path(cpp.__file__).parent
LEGACY_SYMBOLS = {
    "MIGRATED_COMPAT_SYMBOLS",
    "PYTHON_COMPAT_SYMBOLS",
    "DEFERRED_NATIVE_SYMBOLS",
    "_".join(("wcs", "pixel_to_world", "fused", "cpu")),
    "_".join(("wcs", "world_to_pixel", "fused", "cpu")),
    "wcs_zenithal_project",
    "wcs_zenithal_deproject",
    "wcs_cylindrical_project",
    "wcs_cylindrical_deproject",
    "wcs_hpx_project",
    "wcs_hpx_deproject",
    "wcs_tpv_distort",
    "wcs_tpv_invert",
    "healpix_ring2nest_torch_cpu",
    "healpix_nest2ring_torch_cpu",
    "healpix_ang2pix_ring_torch_cpu",
    "healpix_ang2pix_nested_torch_cpu",
    "healpix_pix2ang_ring_torch_cpu",
    "healpix_pix2ang_nested_torch_cpu",
    "healpix_neighbors_ring_torch_cpu",
    "healpix_neighbors_nested_torch_cpu",
    "healpix_get_interp_weights_ring_torch_cpu",
    "healpix_get_interp_weights_nested_torch_cpu",
    "healpix_get_interp_val_ring_torch_cpu",
    "healpix_get_interp_val_nested_torch_cpu",
    "healpix_query_disc_torch_cpu",
    "_sphere_planar_point_in_polygon_cpu",
    "_sphere_arc_intersections_matrix_cpu",
    "_sparse_get_values_blocks_cpu",
    "_healpix_scalar_alm2map_direct_cpu",
    "_healpix_ring_fourier_modes_cpu",
    "_healpix_ring_fourier_modes_spin_conj_cpu",
    "_healpix_ring_fourier_synthesis_cpu",
    "_healpix_spin_ring_finalize_cpu",
    "_healpix_spin_map2alm_ring_concat_cpu",
    "_healpix_spin_interpolate_concat_cpu",
    "_healpix_spin_integrate_concat_cpu",
    "_healpix_spin_integrate_recurrence_cpu",
    "_healpix_spin_interpolate_recurrence_cpu",
    "_healpix_spin_map2alm_ring_recurrence_cpu",
    "_healpix_spin_alm2map_ring_recurrence_cpu",
    "_healpix_scalar_integrate_recurrence_cpu",
    "_healpix_scalar_interpolate_recurrence_cpu",
    "_healpix_spin_map2alm_from_basis_cpu",
    "_healpix_spin_alm2map_from_basis_cpu",
}


def test_removed_symbols_are_not_exposed() -> None:
    assert cpp.__all__ == ["selected_cpu_isa"]
    assert not LEGACY_SYMBOLS.intersection(dir(cpp))


def test_extension_exports_only_registered_kernels() -> None:
    from torchsky import _native_ops

    exported = {name for name in dir(cpp._ext) if not name.startswith("__")}
    # The torch-free core exports one raw-buffer kernel per torch.ops schema
    # registered from Python, the torch-free crossmatch kernels, the ISA
    # diagnostic and the native ABI marker; nothing else.
    expected = set(_native_ops._SCHEMAS) | {
        "NATIVE_ABI",
        "selected_cpu_isa",
        "xmatch_best",
        "xmatch_bp_pairwise_solve",
        "xmatch_bp_pairwise_sweep",
        "xmatch_count",
        "xmatch_fill",
        "xmatch_knn",
        "xmatch_segment_lse_excl",
    }
    assert exported == expected
    bindings = (PACKAGE_ROOT / "cpp_src" / "bindings.cpp").read_text(encoding="utf-8")
    assert all(f'"{name}"' not in bindings for name in LEGACY_SYMBOLS)


def test_native_core_does_not_use_libtorch() -> None:
    # One build serves every PyTorch version: no libtorch headers or symbols.
    sources = [
        p
        for p in (PACKAGE_ROOT / "cpp_src").rglob("*")
        if p.suffix in {".cpp", ".h"} and "third_party" not in p.parts
    ]
    offenders = [
        str(p.relative_to(PACKAGE_ROOT))
        for p in sources
        if any(
            token in p.read_text(encoding="utf-8")
            for token in (
                "#include <torch/",
                "#include <ATen/",
                "#include <c10/",
                "TORCH_LIBRARY",
            )
        )
    ]
    assert offenders == []
    cmake = (PACKAGE_ROOT / "cpp_src" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "find_package(Torch" not in cmake


def test_loader_reload_and_disable_cpp_subprocess_contract() -> None:
    assert importlib.reload(cpp).__all__ == ["selected_cpu_isa"]
    script = """
import sys
import torchsky._native as native
assert native.cpp is None
assert 'torchsky._cpp' not in sys.modules
assert 'torchsky._cpp_ext' not in sys.modules
from torchsky import _ops
assert not _ops.has_fixed_dispatcher()
assert not _ops.has_spectral_dispatcher()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PACKAGE_ROOT.parents[1],
        env={**os.environ, "TORCHSKY_DISABLE_CPP": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_production_python_has_no_removed_cpp_calls() -> None:
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=os.fspath(path))
        legacy_attributes = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in LEGACY_SYMBOLS
        }
        if legacy_attributes:
            relative = path.relative_to(PACKAGE_ROOT)
            offenders.append(f"{relative}: {sorted(legacy_attributes)}")
    assert offenders == []
