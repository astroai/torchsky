## 2024-05-24 - Grouping redundant observations in mapmaking
**Learning:** `_mixture_mc_samples` processes large lists of heterogeneous `FieldObservation` instances, often doing redundant O(N) operations (computing bandpass responses) when the number of unique band configurations (e.g. filters) is actually very small compared to the total number of visits.
**Action:** Always check if operations iterating over many observations can be grouped by bandpass configuration or unique headers (`_observation_band_key`) to drastically reduce redundant continuous evaluations on large proposal grids.
