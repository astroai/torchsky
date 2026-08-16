from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_canfar_wcs_sphere_runner_contract() -> None:
    runner = (ROOT / "scripts" / "run_canfar_wcs_sphere.sh").read_text(encoding="utf-8")
    assert "TORCHSKY_COMMIT" in runner
    assert "wcs-sphere-acceptance-full" in runner
    assert "CONTAINER_IMAGE must name an AstroAI image" in runner
    assert "unknown mode" in runner
    assert "-k cuda" in runner
    assert "--wcs-device cuda" in runner
    assert "vos:sfabbro/torchsky-runs" in runner
    assert "bench_sphere_spectral.py" in runner
    assert "vcp" in runner


def test_canfar_wcs_sphere_launcher_payload_has_no_shell_vars() -> None:
    launcher = (ROOT / "scripts" / "launch_canfar_wcs_sphere.sh").read_text(
        encoding="utf-8"
    )
    assert "canfar" in launcher
    assert "tr ' ' '\\t'" in launcher
    assert "vos:sfabbro/torchsky-runs" in launcher
    assert "refusing canfar payload" in launcher
    fetch = (ROOT / "scripts" / "fetch_canfar_wcs_sphere_run.sh").read_text(
        encoding="utf-8"
    )
    assert "vos:sfabbro/torchsky-runs" in fetch
    assert "vcp" in fetch
