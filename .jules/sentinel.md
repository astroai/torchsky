## 2025-02-18 - XSS via innerHTML in Viz Frontend
**Vulnerability:** Unescaped variables (e.g. `item.name`, `wcs.ctype1`) were passed directly into `innerHTML` strings in `src/torchsky/viz/static/index.html`.
**Learning:** Even internal visualization tools reading standard file formats (like FITS) are vulnerable to XSS if an attacker provides a crafted file with malicious metadata.
**Prevention:** Always use a utility function to escape HTML special characters before interpolating any dynamic strings into `innerHTML`.
