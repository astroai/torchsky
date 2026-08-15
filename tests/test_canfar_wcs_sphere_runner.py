from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_canfar_wcs_sphere_runner_contract() -> None:
    runner = (ROOT / "scripts" / "run_canfar_wcs_sphere.sh").read_text(encoding="utf-8")
    assert "TORCHSKY_COMMIT" in runner
    assert "wcs-sphere-acceptance-full" in runner
    assert "CONTAINER_IMAGE must name an AstroAI image" in runner
    assert "unknown mode" in runner
    assert "-k cuda" in runner
