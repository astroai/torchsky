from __future__ import annotations

from pathlib import Path
import torch

import torchsky
from torchfits import write
from torchsky.map import (
    Observation,
    OutlierConfig,
    ResampleConfig,
    SkyConfig,
    StackConfig,
    aperture_uncertainty,
    blot,
    coadd,
    coadd_cube,
    coadd_multiband,
    detect_outliers,
    match_sky,
    plan,
    plan_cube,
    resample,
)


def _tan_header(
    shape: tuple[int, int] = (32, 32),
    *,
    crval: tuple[float, float] = (150.0, 2.0),
    cdelt: float = 0.001,
    crpix: tuple[float, float] | None = None,
) -> dict[str, object]:
    h, w = shape
    cx, cy = (w / 2.0 + 0.5, h / 2.0 + 0.5) if crpix is None else crpix
    return {
        "NAXIS": 2,
        "NAXIS1": w,
        "NAXIS2": h,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": float(cx),
        "CRPIX2": float(cy),
        "CRVAL1": float(crval[0]),
        "CRVAL2": float(crval[1]),
        "CDELT1": -float(cdelt),
        "CDELT2": float(cdelt),
    }


def test_in_memory_observation_and_plan_projections_and_frames() -> None:
    hdr1 = _tan_header((24, 24), crval=(150.0, 2.0), cdelt=0.001)
    hdr2 = _tan_header((24, 24), crval=(150.004, 2.003), cdelt=0.001)
    img1 = torch.ones((24, 24), dtype=torch.float32) * 3.0
    img2 = torch.ones((24, 24), dtype=torch.float32) * 5.0

    obs1 = (
        Observation(data=img1, wcs=hdr1, device="cpu")
        if False
        else Observation(data=img1, wcs=hdr1)
    )
    obs2 = Observation(data=img2, wcs=hdr2)

    for proj in ("TAN", "SIN", "ZEA", "ARC", "CEA", "CAR", "AIT", "MOL", "GLS"):
        t = plan([obs1, obs2], projection=proj, rotation_deg=15.0, device="cpu")
        assert t.shape[0] > 0 and t.shape[1] > 0
        assert str(t.header["CTYPE1"]).endswith(f"-{proj}")
        assert isinstance(t.wcs, torchsky.WCS)

    # Cross-frame Galactic target planning and resampling
    t_gal = plan([obs1, obs2], projection="TAN", frame="GALACTIC", device="cpu")
    assert t_gal.header["CTYPE1"] == "GLON-TAN"
    assert t_gal.header["CTYPE2"] == "GLAT-TAN"
    res_gal = coadd([obs1, obs2], t_gal, kernel="bilinear", device="cpu")
    assert res_gal.data.shape == t_gal.shape
    assert bool((res_gal.weight > 0).any())


def test_resample_and_coadd_variance_components_and_weight_types() -> None:
    hdr = _tan_header((20, 20), crval=(10.0, -5.0), cdelt=0.001)
    img1 = torch.full((20, 20), 10.0, dtype=torch.float32)
    img2 = torch.full((20, 20), 14.0, dtype=torch.float32)
    vp1 = torch.full((20, 20), 4.0, dtype=torch.float32)
    vp2 = torch.full((20, 20), 4.0, dtype=torch.float32)
    vr1 = torch.full((20, 20), 1.0, dtype=torch.float32)
    vr2 = torch.full((20, 20), 1.0, dtype=torch.float32)
    dq = torch.zeros((20, 20), dtype=torch.int32)
    dq[0, 0] = 4  # bad bit when good_bits=1

    obs1 = Observation(
        data=img1,
        wcs=hdr,
        weight_map=torch.full((20, 20), 2.0, dtype=torch.float32),
        weight_type="rms",
        mask=dq,
        good_bits=1,
        var_poisson=vp1,
        var_rnoise=vr1,
    )
    obs2 = Observation(
        data=img2,
        wcs=hdr,
        weight_map=torch.full((20, 20), 2.0, dtype=torch.float32),
        weight_type="rms",
        mask=dq,
        good_bits=1,
        var_poisson=vp2,
        var_rnoise=vr2,
    )

    single_res = resample(obs1, hdr, device="cpu")
    assert isinstance(single_res, torchsky.map.MapResult)
    assert single_res.var_poisson is not None
    assert single_res.var_rnoise is not None
    assert float(single_res.weight[0, 0].item()) == 0.0
    torch.testing.assert_close(
        single_res.var_poisson[5, 5], torch.tensor(4.0), atol=1e-5, rtol=0.0
    )

    coadded = coadd([obs1, obs2], hdr, device="cpu")
    assert coadded.var_poisson is not None
    assert coadded.var_rnoise is not None
    # Two equal-weight exposures with var_poisson=4 -> combined var_poisson = (4 + 4)/4 = 2.0
    torch.testing.assert_close(
        coadded.var_poisson[5, 5], torch.tensor(2.0), atol=1e-5, rtol=0.0
    )
    torch.testing.assert_close(
        coadded.var_rnoise[5, 5], torch.tensor(0.5), atol=1e-5, rtol=0.0
    )


def test_new_kernels_and_stacking_modes() -> None:
    hdr = _tan_header((20, 20), crval=(30.0, 10.0), cdelt=0.001)
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 20),
        torch.linspace(-1.0, 1.0, 20),
        indexing="ij",
    )
    base = 2.0 + torch.exp(-4.0 * (x**2 + y**2))
    obs1 = Observation(
        data=base,
        wcs=hdr,
        weight_map=torch.ones_like(base),
        weight_is_inverse_variance=True,
    )
    obs2 = Observation(
        data=base * 2.0,
        wcs=hdr,
        weight_map=torch.ones_like(base),
        weight_is_inverse_variance=True,
    )

    for k in (
        "lanczos2",
        "lanczos4",
        "drizzle_turbo",
        "drizzle_gaussian",
        "drizzle_lanczos2",
        "drizzle_lanczos3",
        "conservative_polygon",
    ):
        res = coadd([obs1], hdr, kernel=k, pixfrac=0.8, device="cpu")
        assert res.data.shape == (20, 20)
        assert bool((res.weight > 0).any())

    res_min = coadd([obs1, obs2], hdr, method="min", device="cpu")
    res_max = coadd([obs1, obs2], hdr, method="max", device="cpu")
    res_sum = coadd([obs1, obs2], hdr, method="sum", device="cpu")
    res_chi2 = coadd([obs1, obs2], hdr, method="chi2", device="cpu")
    torch.testing.assert_close(res_min.data, base, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(res_max.data, base * 2.0, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(res_sum.data, base * 3.0, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(
        res_chi2.data,
        torch.sqrt(0.5 * (base**2 + (2.0 * base) ** 2)),
        atol=1e-5,
        rtol=0.0,
    )


def test_exact_square_drizzle_preserves_shifted_compact_source_flux() -> None:
    target = _tan_header((32, 32), crval=(42.0, -12.0), cdelt=0.001)
    y, x = torch.meshgrid(
        torch.arange(32, dtype=torch.float32),
        torch.arange(32, dtype=torch.float32),
        indexing="ij",
    )
    for shifts in (
        ((1.0, -1.0),),
        ((0.25, -0.4),),
        ((-0.25, 0.4), (0.25, -0.4)),
    ):
        observations = []
        for dx, dy in shifts:
            radius_squared = (x - (15.5 + dx)) ** 2 + (y - (15.5 + dy)) ** 2
            image = torch.where(
                radius_squared <= 16.0,
                torch.exp(-0.5 * radius_squared / 1.4**2),
                torch.zeros_like(radius_squared),
            )
            image = image / image.sum()
            observations.append(
                Observation(
                    data=image,
                    wcs=_tan_header(
                        (32, 32),
                        crval=(42.0, -12.0),
                        cdelt=0.001,
                        crpix=(16.5 + dx, 16.5 + dy),
                    ),
                    weight_map=torch.ones_like(image),
                    weight_is_inverse_variance=True,
                )
            )

        result = coadd(
            observations,
            target,
            kernel="drizzle_square",
            pixfrac=1.0,
            semantics="flux",
            method="weighted_mean",
            device="cpu",
        )

        torch.testing.assert_close(
            result.data.sum(), torch.tensor(1.0), atol=2.0e-6, rtol=0.0
        )


def test_blot_and_detect_outliers_cosmic_ray_rejection() -> None:
    hdr = _tan_header((24, 24), crval=(45.0, 0.0), cdelt=0.001)
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 24),
        torch.linspace(-1.0, 1.0, 24),
        indexing="ij",
    )
    true_sky = 1.0 + 5.0 * torch.exp(-8.0 * (x**2 + y**2))

    img1 = true_sky.clone()
    img2 = true_sky.clone()
    img3 = true_sky.clone()
    # Inject a bright cosmic ray and softer adjacent tail in exposure 1
    img1[12, 12] += 200.0
    img1[12, 13] += 35.0
    # And in exposure 2 at a different location
    img2[6, 6] += 180.0

    obs_list = [
        Observation(
            data=img,
            wcs=hdr,
            var_poisson=torch.ones_like(true_sky) * 0.25,
            var_rnoise=torch.ones_like(true_sky) * 0.25,
        )
        for img in (img1, img2, img3)
    ]

    cr_masks, clean_obs = detect_outliers(
        obs_list,
        config=OutlierConfig(snr=(5.0, 3.0), scale=(1.2, 0.7)),
        device="cpu",
    )
    assert bool(cr_masks[0][12, 12].item())
    assert bool(cr_masks[0][12, 13].item())
    assert bool(cr_masks[1][6, 6].item())
    assert not bool(cr_masks[2].any().item())

    # Full coadd with OutlierConfig enabled rejects the cosmic rays automatically
    clean_coadd = coadd(
        obs_list,
        hdr,
        outliers=OutlierConfig(enabled=True, snr=(5.0, 3.0), scale=(1.2, 0.7)),
        stack=StackConfig(method="weighted_mean"),
        device="cpu",
    )
    torch.testing.assert_close(clean_coadd.data, true_sky, atol=1e-4, rtol=1e-4)

    blotted = blot(clean_coadd, obs_list[0], kernel="lanczos3", device="cpu")
    assert isinstance(blotted, torch.Tensor)
    torch.testing.assert_close(
        blotted[4:-4, 4:-4], true_sky[4:-4, 4:-4], atol=1e-2, rtol=1e-2
    )


def test_match_sky_constant_planar_and_per_exposure() -> None:
    hdr = _tan_header((32, 32), crval=(60.0, -10.0), cdelt=0.001)
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 32),
        torch.linspace(-1.0, 1.0, 32),
        indexing="ij",
    )
    signal = 2.0 * torch.exp(-10.0 * (x**2 + y**2))
    obs1 = Observation(data=signal + 10.0, wcs=hdr)
    obs2 = Observation(data=signal + 25.0, wcs=hdr)

    matched_const, info_const = match_sky([obs1, obs2], mode="constant", device="cpu")
    assert info_const["mode"] == "constant"
    torch.testing.assert_close(
        info_const["offsets"], torch.tensor([0.0, 15.0]), atol=1e-4, rtol=0.0
    )
    res_const = coadd(matched_const, hdr, device="cpu")
    torch.testing.assert_close(res_const.data, signal + 10.0, atol=1e-4, rtol=0.0)

    # Planar background gradient difference between obs1 and obs2
    plane_diff = 12.0 + 3.0 * x - 2.0 * y
    obs_p1 = Observation(data=signal, wcs=hdr)
    obs_p2 = Observation(data=signal + plane_diff, wcs=hdr)
    res_planar = coadd(
        [obs_p1, obs_p2],
        hdr,
        sky=SkyConfig(mode="overlap_planar"),
        device="cpu",
    )
    torch.testing.assert_close(res_planar.data, signal, atol=1e-3, rtol=1e-3)


def test_scamp_sidecar_head_and_multiband_cube_and_aperture(tmp_path: Path) -> None:
    base_hdr = _tan_header((16, 16), crval=(100.0, 0.0), cdelt=0.001)
    fits_path = tmp_path / "exp1.fits"
    head_path = tmp_path / "exp1.head"
    img = torch.ones((16, 16), dtype=torch.float32) * 7.0
    write(str(fits_path), img, header=base_hdr, overwrite=True)

    # Write a SCAMP .head sidecar that shifts CRVAL1 to 100.5
    head_path.write_text(
        "CRVAL1  =                100.5 / SCAMP astrometric solution\n"
        "CRVAL2  =                  0.0 / Declination\n"
        "END\n",
        encoding="utf-8",
    )
    obs_sidecar = Observation(
        path=str(fits_path),
        use_sidecar_head=True,
        wavelength=5000.0,
        weight_map=torch.ones_like(img),
        weight_is_inverse_variance=True,
    )
    t = plan([obs_sidecar], device="cpu")
    assert abs(float(t.header["CRVAL1"]) - 100.5) < 1e-6

    res = coadd([obs_sidecar], t, device="cpu")
    ap = torch.zeros_like(res.data)
    ap[6:10, 6:10] = 1.0
    unc = aperture_uncertainty([obs_sidecar], res, ap, device="cpu")
    assert float(unc.variance.item()) > 0.0

    mb = coadd_multiband([obs_sidecar], t, device="cpu")
    assert mb.data.shape == (1, *t.shape)

    ct = plan_cube([obs_sidecar], spectral_axis=[5000.0], device="cpu")
    cb = coadd_cube([obs_sidecar], ct, device="cpu")
    assert cb.data.shape == (1, *ct.spatial_target.shape)


def test_variance_flux_semantics_photometric_scale_and_stcal_sky_modes() -> None:
    hdr = _tan_header((16, 16), crval=(45.0, 10.0), cdelt=0.001)
    img = torch.ones((16, 16), dtype=torch.float32) * 10.0
    vp = torch.ones((16, 16), dtype=torch.float32) * 4.0

    # With identical WCS (Omega_tgt == Omega_src) and photometric_scale=2.0,
    # calibrated data is 2 * 10 = 20 and calibrated variance is 2^2 * 4 = 16
    obs_flux = Observation(
        data=img,
        wcs=hdr,
        photometric_scale=2.0,
        var_poisson=vp,
    )
    res_flux = coadd(
        [obs_flux],
        hdr,
        resample=ResampleConfig(kernel="bilinear", semantics="flux"),
        device="cpu",
    )
    assert res_flux.var_poisson is not None
    torch.testing.assert_close(
        res_flux.data[4:-4, 4:-4],
        torch.full((8, 8), 20.0),
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        res_flux.var_poisson[4:-4, 4:-4],
        torch.full((8, 8), 16.0),
        atol=1e-3,
        rtol=1e-3,
    )

    # stcal sky modes ("global", "global+match") and photometric_scale consistency
    obs_s1 = Observation(data=img + 5.0, wcs=hdr, photometric_scale=2.0)
    obs_s2 = Observation(data=img + 15.0, wcs=hdr, photometric_scale=2.0)
    matched_gm, info_gm = match_sky([obs_s1, obs_s2], mode="global+match", device="cpu")
    assert info_gm["mode"] == "global+match"
    res_gm = coadd(matched_gm, hdr, device="cpu")
    torch.testing.assert_close(res_gm.data, torch.zeros_like(img), atol=1e-4, rtol=0.0)

    # SkyConfig combined with non-linear stacking methods (sigma_clipped, chi2)
    res_sc = coadd(
        [obs_s1, obs_s2],
        hdr,
        sky=SkyConfig(mode="match"),
        stack=StackConfig(method="sigma_clipped"),
        device="cpu",
    )
    torch.testing.assert_close(
        res_sc.data,
        torch.full_like(img, 30.0),
        atol=1e-4,
        rtol=0.0,
    )

    # OutlierConfig with grow=0 and mode="weight_zero"
    noisy = img.clone()
    noisy[8, 8] = 500.0
    obs_cr = [
        Observation(data=noisy, wcs=hdr, weight_map=torch.ones_like(img)),
        Observation(data=img.clone(), wcs=hdr, weight_map=torch.ones_like(img)),
        Observation(data=img.clone(), wcs=hdr, weight_map=torch.ones_like(img)),
    ]
    cr_masks, updated_obs = detect_outliers(
        obs_cr,
        config=OutlierConfig(
            snr=(5.0, 3.0), scale=(1.2, 0.7), grow=0, mode="weight_zero"
        ),
        device="cpu",
    )
    assert int(cr_masks[0].sum().item()) == 1
    assert updated_obs[0].mask is not None
    assert not bool(updated_obs[0].mask[8, 8].item())
    assert updated_obs[0].weight_map is not None
    assert float(updated_obs[0].weight_map[8, 8].item()) == 0.0
