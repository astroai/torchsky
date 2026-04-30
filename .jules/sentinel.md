## 2025-04-24 - Unescaped User Input in DOM (XSS)
**Vulnerability:** The vanilla JavaScript frontend in `src/torchsky/viz/static/index.html` was directly interpolating `dr.recommended_stretch` (which comes from external FITS headers or API responses) into the DOM using `innerHTML` without sanitization.
**Learning:** `innerHTML` is inherently dangerous when combined with string interpolation of external data. The codebase had an existing `htmlEscape` function, but it was inconsistently applied.
**Prevention:** Always wrap external or untrusted string data with `htmlEscape()` before injecting it into `innerHTML`, or prefer using safer alternatives like `textContent`.

## 2025-02-14 - SQL Injection Risk with Exception Catching in DuckDB
**Vulnerability:** DuckDB SQL queries passed to `con.sql()` were protected against multi-statement injection by catching `duckdb.ParserException` for "Multiple SQL statements". However, this approach still allows single-statement injections like `DELETE FROM fits_table` or `DROP TABLE fits_table`.
**Learning:** Relying on exceptions from `con.sql` is insufficient for verifying read-only intent. `duckdb` exposes `duckdb.extract_statements()` which provides an AST analysis of the query, allowing you to explicitly inspect the number and type of statements.
**Prevention:** Always use `duckdb.extract_statements(query)` to enforce that a query contains exactly one statement of type `duckdb.StatementType.SELECT` or `duckdb.StatementType.EXPLAIN` before executing it.
