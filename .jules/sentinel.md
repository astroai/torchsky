## 2024-06-02 - Fix CFITSIO command injection via pipe syntax
**Vulnerability:** CFITSIO allows command execution if a filename starts or ends with a pipe (`|`). This can lead to arbitrary code execution if a FITS file path is user-controlled and unsanitized.
**Learning:** Raw calls to `fits_open_file` must always be preceded by security checks. While `check_fits_filename_security` was implemented in `src/torchsky/cpp_src/security.h` and used in most places, it was missed in the `read_full_nocache` path.
**Prevention:** Always ensure `torchsky::check_fits_filename_security` is called before any invocation of `fits_open_file`.
