## 2024-05-19 - CFITSIO Command Injection
**Vulnerability:** CFITSIO supports extended filename syntax executing commands if the filename ends/starts with '|'. When file paths aren't validated, users can pass command injection strings (e.g. `|ls -l|`) directly to `fits_open_file`, allowing arbitrary code execution.
**Learning:** Pybind/Nanobind C++ interfaces reading directly from python strings into `fits_open_file` must sanitize their inputs before passing to the C library.
**Prevention:** Enforce input validation either on the Python wrapper or specifically via C++ functions like `check_fits_filename` before ANY low level file opening methods or object instantiation is performed.
