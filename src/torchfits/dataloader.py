def _not_supported(*args, **kwargs):
    raise NotImplementedError("Local torchfits shim does not support dataloaders.")


create_distributed_dataloader = _not_supported
create_ml_dataloader = _not_supported


def get_optimal_dataloader_config(*args, **kwargs):
    return {}
