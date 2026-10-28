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


def test_extension_exports_only_isa_diagnostics() -> None:
    exported = {name for name in dir(cpp._ext) if not name.startswith("__")}
    assert exported == {"selected_cpu_isa"}
    bindings = (PACKAGE_ROOT / "cpp_src" / "bindings.cpp").read_text(encoding="utf-8")
    assert bindings.count("m.def(") == 1
    assert all(f'"{name}"' not in bindings for name in LEGACY_SYMBOLS)


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
assert not _ops.has_wcs_dispatcher()
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
