# `torchsky.viz`

`torchsky.viz` is the optional browser visualization surface for inspecting
decoded images, catalogs, MOC/HiPS-style products, and rendered sky data. FITS
decoding at the application boundary is provided by `torchfits`.

## What This Namespace Is For

Use `torchsky.viz` when you need a local browser viewer around torchsky data
products. It is optional and lives behind the `viz` extra because it uses web
server and image-encoding dependencies.

## Install

```bash
pip install "torchsky[viz]"
```

## Recipes

### Serve A Directory

```bash
torchsky viz --root /path/to/data --port 8822
```

### Use From Python

```python
from torchsky.viz import serve

serve(root="/path/to/data", port=8822)
```

## Operational Notes

- Treat the server as a local/development viewer unless you put it behind your
  own authentication and reverse proxy.
- Keep roots narrow; do not serve an entire home directory by accident.
- The served UI is the committed `static/index.html`; there is no npm build
  step or bundler required.
- Large products should use explicit tiling and bounded caches. FITS
  handle/decode caches belong to `torchfits`.

## Related Guides

- [cli.md](cli.md)
- [io.md](io.md)
- [map.md](map.md)
