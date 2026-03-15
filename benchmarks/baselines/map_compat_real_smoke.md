| case | fastest backend | closest external to torchsky | largest pairwise disagreement |
| --- | --- | --- | --- |
| real_astropy_sip_fixture | reproject_interp:bilinear (0.184 ms) | reproject_interp:bilinear (0) | torchsky:bilinear_vs_reproject_interp:bilinear (0) |
| real_astropy_sip_mef | dfreproject:bilinear (0.254 ms) | reproject_interp:bilinear (0) | torchsky:bilinear_vs_reproject_interp:bilinear (0) |
| real_astropy_sip_mef_shifted_wcs | torchsky:bilinear (1.918 ms) | reproject_interp:bilinear (0) | torchsky:bilinear_vs_reproject_interp:bilinear (0) |
| real_astropy_sip_shifted_wcs | torchsky:bilinear (2.819 ms) | reproject_interp:bilinear (0) | torchsky:bilinear_vs_reproject_interp:bilinear (0) |
