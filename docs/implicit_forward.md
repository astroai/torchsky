# Implicit forward modeling (`torchsky.map.field.implicit_forward`)

This note matches the code in [`src/torchsky/map/field/implicit_forward.py`](../src/torchsky/map/field/implicit_forward.py). It describes how to compose a **shared continuous scene** \(F_\theta\) with **per-observation rendering** (bandpass, WCS, PSF—handled inside [`render_field_observation`](map.md)), optional **spectral kinematics** \(\zeta\), exposure **context** \(c\), and a **modular multi-term loss** \(\sum_s w_s L_s\).

It does **not** replace [`fit_field_model`](field.md); use that for full training loops with curricula and diagnostics. Use `implicit_forward` when you want a **small, explicit** scaffold: predict means, apply Charbonnier (robust L1-like) losses with masks/weights, and sum over surveys or instruments.

## 1. Scene and rendering

- **Scene:** any `torch.nn.Module` that implements the field API expected by [`render_field_observation`](map.md) (same contract as [`make_field_model`](field.md)).
- **Observation:** [`FieldObservation`](field.md) — 2D image tensor, FITS-like `header` dict for WCS, spectral band limits or tabulated throughput, optional `weight`, `mask`, `PSFSpec`, backgrounds.

Rendering always goes through `render_field_observation(model, observation, ...)`.

## 2. Context `ForwardContext`

[`ForwardContext`](../src/torchsky/map/field/types.py) labels an exposure for models that accept **`forward_context`** in `forward` / `render_bandpass`:

- `survey`, `instrument`, `detector`, `exposure_id`, `time_mjd`, plus `extra: dict`.

Use this when one shared field is conditioned on instrument or survey (e.g. HST vs JWST). Models that do not take `forward_context` ignore it.

## 3. Kinematics `KinematicState`

[`KinematicState`](../src/torchsky/map/field/implicit_forward.py) holds optional **`one_plus_z`**: observed-to-rest wavelength ratio

\[
\lambda_{\mathrm{obs}} = (1+z)\,\lambda_{\mathrm{emit}} .
\]

This is a **user-chosen linear map** on the spectral axis (no embedded cosmology).

When **`rest_frame_scene=True`** in [`predict_observation_mean`](../src/torchsky/map/field/implicit_forward.py), spectral metadata on the observation is scaled by [`scale_observation_for_rest_frame_scene`](../src/torchsky/map/field/implicit_forward.py) so a scene stored in **rest-frame** \(\lambda_{\mathrm{emit}}\) is queried correctly for **observed** bands.

## 4. Survey terms `SurveyDataTerm`

[`SurveyDataTerm`](../src/torchsky/map/field/implicit_forward.py) groups:

- **`name`** — label for logging.
- **`observations`** — tuple of `FieldObservation`.
- **`loss_weight`** — scalar \(w_s\) multiplying that term’s contribution.
- **`kinematics_per_observation`** — optional per-row `KinematicState | None` (same length as `observations`).
- **`forward_context_per_observation`** — optional per-row `ForwardContext | None`.

Each term contributes \(\sum_i w_s\, L(\mathrm{pred}_i, \mathrm{obs}_i)\) after prediction.

## 5. Prediction API

- **`predict_observation_mean(...)`** — wraps `render_field_observation` with optional rest-frame scaling and `forward_context`.

Additional keyword arguments are forwarded to `render_field_observation` (e.g. `num_spectral_samples`, or **`exposure_pose`** when using bundle-style alignment—see [field.md](field.md)).

## 6. Loss API

- **`observation_charbonnier_loss(prediction, observation, eps=...)`** — Charbonnier \(\rho\) on residuals; combines **`observation.mask`** and **`observation.weight`** (ivar-style weights are typical).

- **`multi_survey_charbonnier_loss(model, terms, rest_frame_scene=False, charbonnier_eps=..., **render_kw)`** — sums weighted losses over all observations in all `SurveyDataTerm`s.

For plain MSE or hooks inside `fit_field_model`, see [field.md](field.md).

## 7. Relation to `fit_field_model`

| Piece | `implicit_forward` | `fit_field_model` |
|--------|-------------------|-------------------|
| Optimizer / schedules | You write it | Built-in |
| Loss choice | Charbonnier scaffold | `FieldLossKind` (mse, charbonnier, huber, gaussian_nll, …) |
| Multi-survey | `SurveyDataTerm` list | Observations list + contexts |
| Best for | Custom training, paper experiments | Standard field fits |

## See also

- [field.md](field.md) — `FieldObservation`, fitting, rendering cubes.
- [map.md](map.md) — map products and field exports.
