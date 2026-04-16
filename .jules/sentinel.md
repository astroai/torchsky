## 2025-02-27 - [PyTorch Insecure Deserialization via torch.load]
**Vulnerability:** Found `torch.load` calls setting `weights_only=False` explicitly, which allows for arbitrary code execution via insecure deserialization during checkpoint loading.
**Learning:** PyTorch models and state dictionaries can contain arbitrary serialized Python objects. By default, legacy PyTorch permits `weights_only=False`. The codebase had explicit `weights_only=False` preventing the safe loading of just tensor weights.
**Prevention:** Always enforce `weights_only=True` when using `torch.load` to prevent arbitrary code execution, especially when loading models from potentially untrusted sources or temporary paths.
