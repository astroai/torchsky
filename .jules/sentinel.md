## 2026-04-18 - Insecure Deserialization in PyTorch Checkpoints
**Vulnerability:** `torch.load` was called with `weights_only=False` in `load_strainer_trunk_checkpoint` (`src/torchsky/map/field/core.py`), allowing arbitrary code execution if a malicious checkpoint file were loaded.
**Learning:** PyTorch defaults changed recently, but older code might explicitly specify `weights_only=False` or omit it. The application was deserializing untrusted files, risking RCE.
**Prevention:** Always enforce `weights_only=True` when using `torch.load` to ensure only tensors and basic types are deserialized, preventing arbitrary object instantiation.
