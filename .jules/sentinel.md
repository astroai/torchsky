## 2026-04-19 - CFITSIO Command Injection Vulnerability
**Vulnerability:** CFITSIO supports extended filename syntax executing commands if the filename ends/starts with '|'. This validation was missing in certain unmapped read paths.
**Learning:** The validation existed in the `FITSFile` constructor but was missed on the lower-level `fits_open_file` paths. Security checks must be applied at the lowest common denominator or universally across all entry points.
**Prevention:** Extract security checks into reusable inline helper functions and apply them consistently before sensitive external library calls.
