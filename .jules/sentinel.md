## 2026-04-18 - Insecure Deserialization in PyTorch Checkpoints
**Vulnerability:** `torch.load` was called with `weights_only=False` in `load_strainer_trunk_checkpoint` (`src/torchsky/map/field/core.py`), allowing arbitrary code execution if a malicious checkpoint file were loaded.
**Learning:** PyTorch defaults changed recently, but older code might explicitly specify `weights_only=False` or omit it. The application was deserializing untrusted files, risking RCE.
**Prevention:** Always enforce `weights_only=True` when using `torch.load` to ensure only tensors and basic types are deserialized, preventing arbitrary object instantiation.

## 2026-04-18 - Internal tests enforce backend boundary checks
**Vulnerability:** Not a direct vulnerability, but a test failure enforcing internal project dependencies and avoiding astropy or numpy imports from `src/torchsky`
**Learning:** `tests/test_no_external_fits_backends.py` requires all code within `src/torchsky` not to depend on `numpy`, `fitsio` or `astropy`. When editing codebase files we must use pure PyTorch APIs (`torch.*`).
**Prevention:** Always use `torch` instead of `numpy` for data manipulation in the `torchsky` source codebase to respect package bounds.
