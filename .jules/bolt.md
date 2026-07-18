## 2024-04-24 - O(1) Header Parsing for FITS Tables
**Learning:** FITS files headers can contain thousands of cards. Iterating through all header items via `for key, value in header_map.items():` for prefixes like `TTYPE`, `TFORM`, and `TNULL` is O(N) and slow since it executes for every single card, even standard metadata cards.
**Action:** When extracting table schema column data from a header (where `Header` inherits directly from `dict`), always try to read the `TFIELDS` key first to get the number of columns, and then use O(1) indexed lookups (e.g., `header_map.get(f"TTYPE{i}")`) for a significant speedup. Ensure fallback logic handles missing or invalid `TFIELDS` gracefully.

## 2024-04-28 - Replace redundant generator comprehension for tensor.shape to tuple casts
**Learning:** For tensor or numpy array shapes (e.g., `tensor.shape`), developers frequently used explicit comprehensions like `tuple(int(v) for v in tensor.shape)`. This creates an overhead through generator evaluation and integer casting. Since `.shape` properties already return an iterable sequence of integers, `tuple(tensor.shape)` is functionally equivalent and over 3.5x faster in tight inner loop evaluations.
**Action:** Always prefer direct tuple wrapping of `.shape` instead of explicit inner loop comprehension casts to integers.

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

## 2024-05-18 - _batch_to_device Fast Path vs Safe Path
**Learning:** Checking for uniform tensor shapes in Python before passing to `torch.stack()` is roughly 3-4x faster when shapes are heterogeneous than handling the exception `torch.stack()` throws. For homogenous shapes, the shape check is about 20% slower than a direct `try/except` around `torch.stack()`. Given that reading HDUs/cutouts frequently yields heterogeneous shapes, prioritizing the safe explicit shape check is better than utilizing exceptions for flow control on the hot path.
**Action:** Always prefer explicitly validating assumptions in Python instead of using exceptions for control flow when the exception pathway is frequently hit, especially when wrapping PyTorch operations.

## 2024-05-30 - Redundant list comprehensions with tensor.tolist()
**Learning:** When calling `.tolist()` on a PyTorch float tensor, it already returns a native Python list of floats. Wrapping this in an explicit list comprehension like `[float(value) for value in tensor.tolist()]` is functionally redundant and introduces unnecessary iteration and casting overhead.
**Action:** Directly assign the result of `tensor.tolist()` instead of using a list comprehension when a list of native Python floats is needed.
