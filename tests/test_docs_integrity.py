from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docs_reference_existing_local_files() -> None:
    expected_paths = [
        "docs/api.md",
        "docs/benchmarks.md",
        "docs/catalogs.md",
        "docs/changelog.md",
        "docs/cli.md",
        "docs/compatibility.md",
        "docs/coordinates.md",
        "docs/encoders.md",
        "docs/examples.md",
        "docs/getting-started.md",
        "docs/install.md",
        "docs/io.md",
        "docs/map.md",
        "docs/ml.md",
        "docs/native-likelihood.md",
        "docs/psf-response-conventions.md",
        "docs/README.md",
        "docs/release.md",
        "docs/spectral.md",
        "docs/table.md",
        "docs/viz.md",
        "docs/wcs.md",
        "docs/roadmap/status.md",
        "docs/roadmap/history.md",
        "docs/projects/field_response_discovery_plan.md",
        "docs/projects/field_response_discovery_status.md",
        "docs/projects/field_response_discovery_amendments.md",
        "docs/archive/torchfits-migration-history.md",
        "docs/archive/roadmap-knowledge-base.md",
        "docs/archive/roadmap-world-model-whitepaper-plan.md",
        "docs/archive/roadmap-sky-measure-scene-plan.md",
        "examples/example_healpix.py",
        "examples/example_image.py",
        "examples/example_image_cube.py",
        "examples/example_image_cutouts.py",
        "examples/example_image_dataset.py",
        "examples/example_image_mef.py",
        "examples/example_polars.py",
        "examples/example_spectral_sht.py",
        "examples/example_table.py",
        "examples/example_table_interop.py",
        "examples/example_table_recipes.py",
        "examples/example_wcs_transform.py",
    ]

    missing = [path for path in expected_paths if not (ROOT / path).exists()]
    assert not missing, f"Missing doc-referenced files: {missing}"
