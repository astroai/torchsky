import re

with open('src/torchsky/__init__.py', 'r') as f:
    content = f.read()

# Only fix _read_batch_paths and _read_batch_hdus
content = content.replace("hdu_batch = _autodetect_hdu(path[0], handle_cache_capacity)", "hdu_batch = _autodetect_hdu(path[0], options.handle_cache_capacity)")
content = content.replace("if mmap is not False:", "if options.mmap is not False:")
content = content.replace("if device != \"cpu\":\n                data_list = _batch_to_device(data_list, device)", "if options.device != \"cpu\":\n                data_list = _batch_to_device(data_list, options.device)")

content = content.replace("effective_mmap = True if isinstance(mmap, str) else mmap", "effective_mmap = True if isinstance(options.mmap, str) else options.mmap")
content = content.replace("if device != \"cpu\":\n            data = _batch_to_device(data, device)", "if options.device != \"cpu\":\n            data = _batch_to_device(data, options.device)")

with open('src/torchsky/__init__.py', 'w') as f:
    f.write(content)
