# Phase 0: WCS and sphere benchmark baseline

Run this **before** and **after** any WCS or `torchsky.sphere` / HEALPix spectral change. Use the **same** Pixi environment and seeds so CSV/JSON rows are comparable.

For **all** wired open-source comparators (Astropy, healpy, hpgeom, legacy PyAST/Kapteyn, map tools, …), environments, and install commands, read **[`docs/benchmarks.md`](../docs/benchmarks.md)**. After bootstrapping optional deps, run `pixi run bench-comparator-versions-all` and archive the emitted JSON next to your `results.csv`.

## 1. Record software versions

From the repo root (`torchsky/`):

```bash
pixi run -e bench-all python - <<'PY'
import json, os, subprocess, sys
from pathlib import Path

def ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", str(m))
    except Exception as e:
        return f"missing:{e}"

pkgs = {
    "python": sys.version.split()[0],
    "torch": ver("torch"),
    "numpy": ver("numpy"),
    "astropy": ver("astropy"),
    "healpy": ver("healpy"),
    "hpgeom": ver("hpgeom"),
    "astropy_healpix": ver("astropy_healpix"),
    "healsparse": ver("healsparse"),
}
try:
    import torchsky
    pkgs["torchsky"] = getattr(torchsky, "__version__", "dev")
except Exception as e:
    pkgs["torchsky"] = f"missing:{e}"

sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
env = {k: os.environ[k] for k in sorted(os.environ) if k.startswith("TORCHSKY_")}
run_id = "phase0_manual"
out_dir = Path("benchmarks_results") / run_id
out_dir.mkdir(parents=True, exist_ok=True)
Path(out_dir / "VERSIONS.json").write_text(
    json.dumps({"git_sha": sha, "packages": pkgs, "TORCHSKY_env": env}, indent=2) + "\n",
    encoding="utf-8",
)
print(Path(out_dir / "VERSIONS.json").read_text())
PY
```

## 2. Editable install (required for `import torchsky` in benchmark drivers)

```bash
pixi run -e bench-all pip install -e . --no-build-isolation
```

## 3. WCS and sphere contract runs

Quick smoke (small `N`, subset of projections / sphere cases):

```bash
RUN_ID=phase0_wcs_sphere_$(date +%Y%m%d)
pixi run -e bench-all python benchmarks/bench_all.py --profile user --wcs-only --quick --run-id "$RUN_ID" --output-dir benchmarks_results
pixi run -e bench-all python benchmarks/bench_all.py --profile user --sphere-only --quick --run-id "${RUN_ID}_sphere" --output-dir benchmarks_results
```

Spin parity and throughput (optional gates; use `--disable-gates` for reporting only):

```bash
pixi run -e sphere-bench python benchmarks/bench_spin_matrix.py --case-set default --runs 3 --json-out benchmarks_results/"$RUN_ID"/spin_matrix.json
```

Optional full-domain runs (longer): omit `--quick` and use `pixi run bench` / `pixi run bench-wcs` / `pixi run bench-sphere` from [`pixi.toml`](../pixi.toml).

## 4. Artifacts

Under `benchmarks_results/<run_id>/`:

| File | Content |
|------|---------|
| `results.csv` | Per-row timings and comparators |
| `torchsky_deficits.csv` | Rows where torchsky is not first |
| `summary.md` | Human-readable summary |
| `VERSIONS.json` | Git SHA, package versions, `TORCHSKY_*` env |
| `spin_matrix.json` | healpy spin matrix output (if run) |

## 5. Post-change iteration

Re-run sections 3–4 with a **new** `run_id`, then diff `torchsky_deficits.csv` and spin JSON against the Phase 0 baseline.

## 6. One-shot acceptance (benches + pytest + spin matrix)

From repo root, after `pip install -e .` in `bench-all`:

```bash
pixi run wcs-sphere-acceptance
```

This runs quick WCS and sphere `bench_all` rows (`acceptance_wcs` / `acceptance_sphere` under `benchmarks_results/`), then **all** `tests/test_wcs_*.py` and `tests/test_sphere_*.py`, then `benchmarks/bench_spin_matrix.py --case-set default --runs 2` and writes `benchmarks_results/acceptance_spin_matrix.json`. Inspect that JSON for median `torchsky/healpy` ratios.
