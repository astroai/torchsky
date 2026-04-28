## 2025-04-24 - Unescaped User Input in DOM (XSS)
**Vulnerability:** The vanilla JavaScript frontend in `src/torchsky/viz/static/index.html` was directly interpolating `dr.recommended_stretch` (which comes from external FITS headers or API responses) into the DOM using `innerHTML` without sanitization.
**Learning:** `innerHTML` is inherently dangerous when combined with string interpolation of external data. The codebase had an existing `htmlEscape` function, but it was inconsistently applied.
**Prevention:** Always wrap external or untrusted string data with `htmlEscape()` before injecting it into `innerHTML`, or prefer using safer alternatives like `textContent`.
