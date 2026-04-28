## 2024-04-24 - O(1) Header Parsing for FITS Tables
**Learning:** FITS files headers can contain thousands of cards. Iterating through all header items via `for key, value in header_map.items():` for prefixes like `TTYPE`, `TFORM`, and `TNULL` is O(N) and slow since it executes for every single card, even standard metadata cards.
**Action:** When extracting table schema column data from a header (where `Header` inherits directly from `dict`), always try to read the `TFIELDS` key first to get the number of columns, and then use O(1) indexed lookups (e.g., `header_map.get(f"TTYPE{i}")`) for a significant speedup. Ensure fallback logic handles missing or invalid `TFIELDS` gracefully.

## 2024-04-28 - Replace redundant generator comprehension for tensor.shape to tuple casts
**Learning:** For tensor or numpy array shapes (e.g., `tensor.shape`), developers frequently used explicit comprehensions like `tuple(int(v) for v in tensor.shape)`. This creates an overhead through generator evaluation and integer casting. Since `.shape` properties already return an iterable sequence of integers, `tuple(tensor.shape)` is functionally equivalent and over 3.5x faster in tight inner loop evaluations.
**Action:** Always prefer direct tuple wrapping of `.shape` instead of explicit inner loop comprehension casts to integers.
