## 2026-04-12 - [XSS in HDUList HTML Representation]
**Vulnerability:** Cross-Site Scripting (XSS) vulnerability found in `HDUList._repr_html_` method inside `src/torchsky/hdu.py`.
**Learning:** `EXTNAME` header fields and other properties obtained from arbitrary FITS files were passed directly into string interpolation formatting an HTML table without escaping malicious tags. When working with Jupyter notebook integrations, users may accidentally execute JavaScript simply by attempting to print the `HDUList` associated with an untrusted `.fits` file.
**Prevention:** Always use `html.escape()` before directly interpolating untrusted variables parsed from files into HTML output, avoiding shadowing the `html` standard library module name when defining lists.
