## 2025-04-15 - [CRITICAL] Fix insecure deserialization in field checkpoints
**Vulnerability:** Found `torch.load(..., weights_only=False)` in `src/torchsky/map/field/core.py` (load_strainer_trunk_checkpoint).
**Learning:** PyTorch default behavior changed but legacy loaders explicitly passed `weights_only=False` which allowed arbitrary code execution.
**Prevention:** Always enforce `weights_only=True` for torch.load unless absolutely necessary and documented, especially for external models.
