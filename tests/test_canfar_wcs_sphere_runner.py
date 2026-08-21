from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_canfar_wcs_sphere_runner_contract() -> None:
    runner = (ROOT / "scripts" / "run_canfar_wcs_sphere.sh").read_text(encoding="utf-8")
    assert "TORCHSKY_COMMIT" in runner
    assert "acceptance_wcs_full" in runner
    assert "--no-legacy-wcs-bridge" in runner
    assert "CONTAINER_IMAGE must name an AstroAI image" in runner
    assert "unknown mode" in runner
    assert runner.index("python -m pip install astropy") < runner.index("-k cuda")
    assert "--wcs-device cuda" in runner
    assert "vos:sfabbro/torchsky-runs" in runner
    assert "bench_sphere_spectral.py" in runner
    assert "vcp" in runner
    assert "gist create" in runner
    assert "trap on_exit EXIT" in runner
    assert "vos_upload heartbeat" in runner
    assert 'vmkdir -p "${vos_dest}" 2>/dev/null || true' not in runner
    assert "cuda_rc" in runner
    assert "cpu_rc" in runner
    assert "cuda job failures" in runner
    assert "cpu job failures" in runner
    assert runner.index("-k cuda") < runner.index("acceptance_wcs_cuda")
    assert runner.index("--no-legacy-wcs-bridge") < runner.index("acceptance_wcs_cuda")
    assert "vos_upload after_wcs" in runner
    for mode in ("spectral", "sphere", "sparse", "spin", "geometry", "cuhpx"):
        assert f"\n  {mode})" in runner or f"\n  {mode} |" in runner
    assert "sphere_spectral_nside128.json" in runner
    assert "bench_sphere_sparse.py" in runner
    assert "bench_cuhpx_sht_spike.py" in runner
    assert "--no-throughput-gate" in runner
    assert runner.count("--no-throughput-gate") >= 4
    assert "--n-queries 200000" in runner
    assert "--n-queries 500000" in runner
    for mode in (
        "wcs-paper2",
        "wcs-tpv",
        "wcs-spectral",
        "wcs-tabular",
        "sht-profile",
        "moc",
        "catalog",
        "observed",
        "geom-cuda",
        "world-cuda",
    ):
        assert f"\n  {mode})" in runner
    assert "--moc-only" in runner
    assert "catalog_n1e7.json" in runner
    assert "--skip-decl-pruned" in runner
    assert "test_wcs_observed_frames.py" in runner
    assert "test_catalogs.py" in runner
    assert "bench_wcs_spectral_temporal.py" in runner
    assert "bench_wcs_tabular.py" in runner
    assert "bench_wcs_small_n_paths.py" in runner
    assert "--device cuda" in runner
    assert "wcs_paper2_cuda.json" in runner
    assert "observed_place_cuda.json" in runner
    assert "catalog_cuda.json" in runner


def test_canfar_wcs_sphere_launcher_payload_has_no_shell_vars() -> None:
    launcher = (ROOT / "scripts" / "launch_canfar_wcs_sphere.sh").read_text(
        encoding="utf-8"
    )
    assert "canfar" in launcher
    assert "tr ' ' '\\t'" in launcher
    assert "vos:sfabbro/torchsky-runs" in launcher
    assert "refusing canfar payload" in launcher
    assert "gh repo clone sfabbro/torchsky" in launcher
    assert "git push before launching CANFAR" in launcher
    assert "RUNNER_VOS_URI" not in launcher
    assert "suite" in launcher
    assert "cuhpx" in launcher
    assert "sht" in launcher
    assert "world" in launcher
    assert "world-cuda" in launcher
    assert "wcs-spectral" in launcher
    assert "wcs-tabular" in launcher
    fetch = (ROOT / "scripts" / "fetch_canfar_wcs_sphere_run.sh").read_text(
        encoding="utf-8"
    )
    assert "vos:sfabbro/torchsky-runs" in fetch
    assert "vcp" in fetch
