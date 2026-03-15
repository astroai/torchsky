# Map Truth Pipeline

```mermaid
flowchart TD
    A["Latent sky field on the sphere"] --> B["Choose target product\nbandpass + WCS + union grid"]
    A --> C["Simulate each exposure"]
    C --> C1["Sample sky through source WCS\n(TAN or TPV)"]
    C1 --> C2["Apply PSF blur"]
    C2 --> C3["Apply exposure time and transmission"]
    C3 --> C4["Add noise"]
    C4 --> C5["Write image + per-pixel weight map"]
    C5 --> C6["Optionally write explicit mask plane"]
    B --> D["Final reconstruction methods"]
    C6 --> D
    D --> D1["Torchsky stack\nresample weighted numerator and denominator\nwith distorted-WCS quadrature for flux products,\nthen weighted mean"]
    D --> D2["Legacy stackers\nreproject/dfreproject/drizzle/montage/swarp/casutools"]
    D --> D3["Torchsky field model\nfit continuous field, then render to target product"]
    B --> E["Noiseless target truth product"]
    D1 --> F["Compare reconstructed product to truth"]
    D2 --> F
    D3 --> F
    E --> F
    F --> G["Metrics\nRMSE, MAE, p95/max error,\nflux error, SNR, coverage, zero-support"]
```

The evaluation contract is always the final requested product, not an internal warp.
Each method is judged after producing the same target-grid image or cube.
For the current distorted `SIP`/`TPV` stack path, torchsky approximates target-pixel area integration with fixed 2x2 Gauss-Legendre subpixel quadrature for `flux` products. The default `surface_brightness` product path uses a single target-center sample instead of pixel-area integration.
