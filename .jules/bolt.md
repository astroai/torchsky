
## 2025-05-15 - Cache FITS filename extension check
**Learning:** Checking for CFITSIO extended syntax (presence of '[') in filenames using `std::string::find` is an (N)$ operation that was being performed repeatedly across multiple I/O methods. Since the filename is immutable for the lifetime of the `FITSFile` object, this search is redundant.
**Action:** Cache the result of the extension check in a boolean member variable (`has_extension_`) during object construction and use the cached value in all subsequent I/O related methods.
