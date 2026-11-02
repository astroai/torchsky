"""Crossmatch two large synthetic catalogues (1e6 x 1e6 by default).

Left catalogue: N uniformly distributed sky positions. Right catalogue:
for half of them a counterpart with 0.3 arcsec Gaussian astrometric scatter,
plus uniformly distributed unrelated sources, N rows in total. ``find="best"``
returns the nearest counterpart within 1 arcsec; ``find="all"`` returns every
pair. Results are checked against the injected truth. Candidates come from
declination strips x sorted right ascension with an exact great-circle
verification, so the cost is O((N + M) log M) rather than O(N*M).
Runtime about 3 s on CPU (about 0.5 s per mode for the match itself).

    pixi run -e test -- python examples/crossmatch_million.py [--n N] [--device cuda]
"""

from __future__ import annotations

import argparse
import time

import torch

from torchsky.catalogs import crossmatch_sky


def uniform_sky(n: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    ra = torch.rand(n, generator=g, dtype=torch.float64) * 360.0
    dec = torch.rad2deg(torch.asin(2.0 * torch.rand(n, generator=g, dtype=torch.float64) - 1.0))
    return ra, dec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    g = torch.Generator().manual_seed(11)
    n = args.n or 1_000_000
    n_true = n // 2

    ra_l, dec_l = uniform_sky(n, g)
    sigma_deg = 0.3 / 3600.0
    jitter = torch.randn(2, n_true, generator=g, dtype=torch.float64) * sigma_deg
    dec_t = (dec_l[:n_true] + jitter[1]).clamp(-90.0, 90.0)
    ra_t = (ra_l[:n_true] + jitter[0] / dec_t.deg2rad().cos().clamp(min=1e-3)) % 360.0
    ra_f, dec_f = uniform_sky(n - n_true, g)
    perm = torch.randperm(n, generator=g)
    ra_r = torch.cat([ra_t, ra_f])[perm]
    dec_r = torch.cat([dec_t, dec_f])[perm]
    truth_of_right = torch.cat([torch.arange(n_true), torch.full((n - n_true,), -1)])[perm]

    dev = args.device
    args_xy = [t.to(dev) for t in (ra_l, dec_l, ra_r, dec_r)]
    for find in ("best", "all"):
        t0 = time.perf_counter()
        res = crossmatch_sky(*args_xy, radius_arcsec=1.0, find=find, device=dev)
        dt = time.perf_counter() - t0
        left, right = res.left_index.cpu(), res.right_index.cpu()
        correct = (truth_of_right[right] == left) & (left < n_true)
        print(f"find={find:4s}: {left.numel():>8d} pairs in {dt:5.1f} s "
              f"({n / dt / 1e6:.2f} M left rows/s); injected pairs recovered: "
              f"{int(correct.sum())}/{n_true} ({correct.sum().item() / n_true:.4f})")
        if find == "best":
            assert correct.sum().item() / n_true > 0.995
            sep = res.separation_arcsec.cpu()
            print(f"          median separation {float(sep[correct].median()):.3f} arcsec "
                  f"(expected {0.3 * 1.1774:.3f} for sigma=0.3 arcsec per axis)")
    print(f"device={dev}")


if __name__ == "__main__":
    main()
