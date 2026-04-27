## 2024-04-10 - [Vectorizing Range Finding]
**Learning:** Finding contiguous ranges in sorted NumPy arrays using Python loops is significantly slower than using vectorized NumPy operations. Utilizing `np.diff` with `np.nonzero` allows calculating range breaks and lengths efficiently, yielding over a 5x speedup for 10k-element arrays while remaining clean and concise. Edge cases involving empty arrays must be explicitly handled.
**Action:** Always prefer NumPy vectorization (like `np.diff` and `np.nonzero`) over raw Python iteration when identifying continuous blocks or calculating intervals within numeric arrays.

## 2025-04-10 - [PyTorch Fourier Features Vectorization]
**Learning:** Iterating over tensors and appending PyTorch operations iteratively to lists (`for scale in scales: feats.append(...)`) is slow. By broadcasting dimensions using `unsqueeze`, we can perform operations simultaneously. Interleaving `sin` and `cos` features can be elegantly done by combining `torch.stack` and `.view()`.
**Action:** Always look for opportunities to replace `for` loops with vectorized PyTorch operations using dimension broadcasting, stacking, and reshaping.

## 2024-05-24 - Vectorize bandpass loop
**Learning:** Replaced loops inside PyTorch feature extractor with broadcasting over scales and then summing over spatial dimensions. Achieved 15x speedup by eliminating Python loops and multiple small tensor allocations.
**Action:** Prefer vectorized broadcasting operations in PyTorch. Avoid using Python loop iterations to append calculated features to lists which is later stacked.

## 2024-05-24 - Faster Squaring
**Learning:** We confirmed that `centered_sq = centered * centered` is faster than `torch.square(centered)`.
**Action:** Always use element-wise multiplication `x * x` instead of `torch.square(x)` or `x ** 2` for better performance.

## 2025-05-15 - Cache FITS filename extension check
**Learning:** Checking for CFITSIO extended syntax (presence of '[') in filenames using `std::string::find` is an O(N) operation that was being performed repeatedly across multiple I/O methods. Since the filename is immutable for the lifetime of the `FITSFile` object, this search is redundant.
**Action:** Cache the result of the extension check in a boolean member variable (`has_extension_`) during object construction and use the cached value in all subsequent I/O related methods.

## 2024-05-18 - FITS Header Dict Lookup Optimization
**Learning:** `torchsky.hdu.Header` inherits directly from Python's built-in `dict`. Consequently, individual lookups (`header.get()`) are O(1) and extremely fast. When extracting specific known key patterns (like `TFORM{i}`, `TUNIT{i}`), attempting to iterate over all `header.items()` manually to parse out integer prefixes actually introduces significant parsing overhead and ends up being much slower (nearly 2x slower) than simply performing the targeted `get()` operations with unrolled loops.
**Action:** When extracting specifically indexed properties from FITS headers in `torchsky`, unroll inner loops and use direct string concatenation (`prefix + str(index)`) coupled with direct `.get()` lookups. Avoid iterating over `header.items()` to "find" indices, and avoid using strict `type(x) is str` checks (which break type inference and polymorphism) in favor of the standard `isinstance(x, str)` when testing header values.
## 2024-05-14 - Optimize FITS Header Parsing
**Learning:** FITS header dictionaries inherit from Python's standard `dict` but can contain massive amounts of metadata cards (e.g. `COMMENT` and `HISTORY` keys). Iterating over all `header.items()` to extract predefined fields like column types causes an unexpected O(N) performance bottleneck during instantiation, scaling poorly on densely documented astronomical files.
**Action:** Always favor targeted dictionary lookups (`header.get(KEY)`) over exhaustive iteration. Specifically for tables, extract `TFIELDS` first and query only the valid column headers in an O(TFIELDS) lookup loop rather than scanning the entire FITS header mapping.
## 2024-05-18 - Optimize Table String Column Inference
**Learning:** `torchsky.hdu.Header` inherits directly from `dict`. Iterating through all `header.items()` to find `TFORM` and `TTYPE` keys is slow, especially for headers with many cards. Instead, parsing `TFIELDS` first and executing O(1) dictionary lookups (`header.get(f"TFORM{i}")`) reduces iteration overhead significantly.
**Action:** Always favor targeted dictionary lookups (`header.get()`) using known key patterns (e.g., indexed from `1` to `TFIELDS`) instead of iterating through `header.items()` when processing FITS headers.
