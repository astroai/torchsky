## 2025-04-24 - Unescaped User Input in DOM (XSS)
**Vulnerability:** The vanilla JavaScript frontend in `src/torchsky/viz/static/index.html` was directly interpolating `dr.recommended_stretch` (which comes from external FITS headers or API responses) into the DOM using `innerHTML` without sanitization.
**Learning:** `innerHTML` is inherently dangerous when combined with string interpolation of external data. The codebase had an existing `htmlEscape` function, but it was inconsistently applied.
**Prevention:** Always wrap external or untrusted string data with `htmlEscape()` before injecting it into `innerHTML`, or prefer using safer alternatives like `textContent`.
## 2024-04-29 - Prevent XSS in FITS metadata visualization via innerHTML
**Vulnerability:** XSS vulnerability where FITS metadata (returned from the backend) like `wcs.crval1` and `dr.dr_ratio` were interpolated into the DOM using `.innerHTML` without HTML escaping.
**Learning:** `innerHTML` allows arbitrary HTML execution if user or file-controlled strings are evaluated. The FITS metadata (like WCS and DR stats) comes directly from files (the `api`/backend parses FITS headers), which an attacker could craft to contain `<script>` tags or malicious HTML.
**Prevention:** Always use `htmlEscape()` when setting `innerHTML` with values derived from external inputs (like FITS file content/metadata) or use `textContent` where applicable to avoid XSS execution.
