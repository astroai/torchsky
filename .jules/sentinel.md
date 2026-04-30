## 2026-04-24 - Secure Default CORS Policy
**Vulnerability:** Insecure Default CORS Policy in `torchsky.viz`.
**Learning:** Defaulting to `["*"]` (wildcard) for CORS `allow_origins` when the user provides `None` or leaves it unconfigured is a dangerous default. It allows any website to make cross-origin requests to the visualization server, potentially leading to CSRF or data exfiltration if other protections are weak.
**Prevention:** Always default to a restrictive state (e.g., an empty list `[]` for allowed origins) in security-sensitive configurations like CORS. Force the user to explicitly opt-in to permissive cross-origin access.
