# Documentation authoring

How to write and change the published documentation. The site is built from
`docs/` with Zensical and must pass `pixi run docs-contract` before a docs pull
request is mergeable.

## Where content belongs

| Content | Home |
|---|---|
| Task-oriented workflows, concepts, recipes, caveats | Domain guide in `docs/` (`wcs.md`, `map.md`, `sphere.md`, …) |
| Signatures, parameters, return contracts, per-name docstrings | Generated reference under `docs/reference/` |
| Page title, stability tier, grouping order, cross-links, quick paths | `docs/reference/_registry.toml` |
| Measured performance tables and parity summaries | `docs/evaluation/` plus its committed data artifacts |
| Historical design notes, audits, research, scratch analysis | `notes/` — never inside `docs/` |
| Release history | `docs/changelog.md` |

Rules of thumb:

- A guide answers *why* and *when*. The reference answers *what exactly*.
- Never copy a signature into a guide. Link to the rendered reference anchor so
  the signature cannot go stale.
- FITS/HDU/table internals belong to `torchfits`. Document only the boundary
  contract here.
- Scientific parity, real-data validation, and scorecards live in
  `torchsky-harness`; link them, do not restate them as new measurements.

Every published page must appear in the `nav` list in `zensical.toml`; the docs
integrity test fails otherwise.

## Reference pages are generated

`docs/reference/*.md` is generated from `docs/api-manifest.json` and
`docs/reference/_registry.toml` by `scripts/gen_reference.py`. Do not edit those
pages by hand.

- Fix a wrong signature or explanation by editing the **docstring or source**,
  not the generated Markdown.
- Change page identity, tier, ordering, grouping labels, or related guides in
  `_registry.toml`.
- Regenerate with `pixi run docs-reference`; verify with
  `pixi run docs-reference-check`.
- `docs/reference/runtime-contract.md` is hand-written; it is the one reference
  page outside the generator.

The generator classifies every public name and handles names that are exported
lazily, aliased, or shadowed by a same-named submodule, so a public name cannot
disappear from the reference without the check failing.

## Docstring standard

Reference quality is docstring quality. Public names use Google-style
docstrings:

```python
def coadd(observations, target, *, resample=None):
    """Coadd calibrated observations onto one planned target grid.

    The value semantics come from ``ResampleConfig`` and are recorded in the
    result provenance; no implicit convention is applied.

    Args:
        observations: Calibrated inputs. Each image must carry a WCS.
        target: Planned target grid from :func:`plan`.
        resample: Resampling and value-semantics configuration.

    Returns:
        ``MapResult`` with ``data``, ``weight``, ``header``, and provenance.

    Raises:
        ValueError: If an observation has no usable celestial WCS.

    Examples:
        >>> result = coadd([obs], target)
        >>> result.data.shape == target.shape
        True
    """
```

- One imperative summary line, then a blank line, then details.
- `Args:` / `Returns:` / `Raises:` / `Examples:` sections; types live in
  annotations, not in the docstring.
- State units, shapes, pixel origin, device behaviour, and value semantics when
  they are part of the contract.
- Examples are short and self-contained; they are checked like documentation
  snippets.
- Do not restate the signature in prose, and do not claim measured performance.

## Code in prose

Runnable, self-contained Python blocks are marked with an HTML comment directly
before the fence:

````markdown
<!-- docs: run -->
```python
import torch
import torchsky.sphere as sphere

pix = sphere.ang2pix(256, torch.tensor([10.0]), torch.tensor([5.0]), lonlat=True)
assert pix.shape == (1,)
```
````

Rules for a marked block:

- Self-contained: it imports everything it uses and needs no FITS file, network,
  or optional extra.
- Cheap and deterministic: small arrays, fixed seeds, a few seconds at most.
- It asserts the outcome it claims, so the docs test catches behavior drift.
- If it needs an optional package (`matplotlib`, `astropy`, `polars`, …), leave
  it unmarked; unmarked fences are still import-checked but not executed.

Unmarked Python fences are validated for import resolution. Prefer real
self-contained inputs over placeholder filenames: a tutorial that cannot be run
is not a tutorial.

## Links and anchors

- Use relative links between published pages: `[WCS guide](wcs.md)`.
- Link to a rendered API entry with its canonical anchor:
  `[`WCS`](reference/wcs.md#torchsky.wcs.WCS)`.
- Never link into `src/`, `notes/`, `examples/` internals, or generated
  `published-examples/` files that do not exist in `examples/`.
- External links are allowed; first-party 404s fail the link check.
- Every page has exactly one `# H1`; the file name supplies the nav label.

## Math, admonitions, and structure

- Use `\(inline\)` and `\[display\]` math; MathJax is configured for the site.
- Use admonitions for operational caveats:
  `!!! warning "Experimental surface"` or `!!! tip "When to use"`.
- Typical page shape: purpose → concepts → decision table → runnable recipes →
  caveats → related guides.
- Title Case section headings; sentence case prose; no duplicated introduction
  and no second H1.
- State the stability tier for experimental surfaces (`scene`, `field`, native
  likelihood/operator modules, `viz` server paths).

## Checks

Run from the repository root:

```bash
pixi run docs-serve                 # live preview
pixi run docs-reference             # regenerate API reference pages
pixi run docs-reference-check       # committed pages are current
pixi run docs-build                 # strict site build
pixi run docs-links                 # local link check on built site
pixi run docs-contract              # full docs gate
pixi run -e test -- pytest tests/test_docs_integrity.py tests/test_docs_snippets.py tests/test_docs_coverage.py -q
pixi run api-manifest-check         # public-name contract unchanged or updated deliberately
```

If a public name is added, removed, or renamed, update `docs/api-manifest.json`
with `pixi run api-manifest`, regenerate the reference, and mention the change
in the guide that introduces the surface. A docs-only change does not need the
full test suite; a change that alters public names does.
