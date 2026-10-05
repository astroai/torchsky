from pathlib import Path
import json
import sys

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import score_field_models as scorer  # noqa: E402


def test_measure_effective_psfs_uses_each_staged_cell(monkeypatch, tmp_path: Path) -> None:
    methods = [("legacy", "grid"), ("footprint", "grid")]
    staged = {cell: tmp_path / "inputs" / cell[0] for cell in methods}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_id": ["bright-a", "faint", "bright-b"],
                "reference_ab_magnitude": [27.0, 28.0, 27.0],
                "world_radec_deg": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                "band_flux_jy": {"F150W": [3.0, 1.0, 2.0]},
            }
        )
    )
    calls = []

    def fake_effective_psf(**kwargs):
        calls.append(kwargs)
        method = kwargs["methods"][0]
        return {"rows": [{"band": "F150W", "method": method}]}

    monkeypatch.setattr(scorer, "_updec_effective_psf_artifact", fake_effective_psf)
    rows = scorer._measure_effective_psfs(
        staged,
        methods,
        ("F150W",),
        manifest,
        tmp_path / "effective_psfs",
    )

    assert [row["method"] for row in rows] == ["legacy__grid", "footprint__grid"]
    assert calls[0]["blank_product_root"] == staged[methods[0]] / "blank"
    assert calls[1]["positive_product_root"] == staged[methods[1]] / "validation-000"
    calibration_manifest = json.loads(
        calls[0]["injection_manifest_path"].read_text()
    )
    assert calibration_manifest["source_count"] == 2
    assert calibration_manifest["source_id"] == ["bright-a", "bright-b"]
    assert calibration_manifest["reference_ab_magnitude"] == [27.0, 27.0]
    assert calibration_manifest["world_radec_deg"] == [[1.0, 2.0], [5.0, 6.0]]
    assert calibration_manifest["band_flux_jy"]["F150W"] == [3.0, 2.0]


def test_m50_uses_the_faintward_completeness_crossing() -> None:
    assert scorer._m50([27.0, 27.5, 28.0, 28.5, 29.0], [1.0, 0.9, 0.75, 0.5, 0.0]) == 28.5
    assert scorer._m50([27.0, 28.0, 29.0], [1.0, 0.75, 0.25]) == 28.5


def test_selection_rejects_a_deeper_method_with_worse_resolution() -> None:
    rows = [
        {
            "method": "control",
            "psf_mode": "control",
            "target_purity_reached": True,
            "heldout_gate_pass": True,
            "resolution_gate_pass": True,
            "m50_ci": {"m50": 28.5},
        },
        {
            "method": "candidate",
            "psf_mode": "candidate",
            "target_purity_reached": True,
            "heldout_gate_pass": True,
            "resolution_gate_pass": False,
            "m50_ci": {"m50": 29.0},
        },
    ]

    winner, _ = scorer._select(rows, calibration_fit_report=None)

    assert winner is rows[0]


def test_best_cell_products_copy_only_the_gated_winner(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    out_dir = tmp_path / "score"
    for psf_mode, value in (("control", b"accepted"), ("candidate", b"rejected")):
        source = (
            input_root
            / "b"
            / "blank"
            / "products"
            / psf_mode
            / "F150W"
            / "grid"
        )
        source.mkdir(parents=True)
        (source / "prediction.png").write_bytes(value)

    scorer._copy_selected_cell_products(
        input_root=input_root,
        out_dir=out_dir,
        module="b",
        bands=("F150W",),
        winner={"psf_mode": "control", "model_kind": "grid"},
    )

    copied = out_dir / "best_cell_products" / "F150W" / "F150W_prediction.png"
    assert copied.read_bytes() == b"accepted"


def test_best_cell_products_are_absent_when_no_method_passes(tmp_path: Path) -> None:
    scorer._copy_selected_cell_products(
        input_root=tmp_path / "inputs",
        out_dir=tmp_path / "score",
        module="b",
        bands=("F150W",),
        winner=None,
    )

    assert not (tmp_path / "score" / "best_cell_products").exists()
