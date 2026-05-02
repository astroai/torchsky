import re

with open('src/torchsky/__init__.py', 'r') as f:
    content = f.read()

content = content.replace("def _read_batch_paths_fake(", "def _read_batch_paths(")
content = content.replace("def _read_batch_hdus_fake(", "def _read_batch_hdus(")
content = content.replace("ReadOptionsFake(", "ReadOptions(")

with open('src/torchsky/__init__.py', 'w') as f:
    f.write(content)
