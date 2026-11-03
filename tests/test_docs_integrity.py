"""Documentation contract: published tree, navigation, assets, and links.

The reference implementation is `torchfits`: a Zensical/Material site whose
published pages are exactly the files under `docs/`, with planning and research
notes kept outside the published tree. These tests keep that boundary and catch
broken relative links before the site build does.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# Pages that must exist because the navigation, README, or domain routing
# depends on them. Keep this list curated rather than exhaustive so moving a
# guide between nav groups does not break the contract test.
REQUIRED_PAGES = (
    "index.md",
    "introduction/index.md",
    "introduction/install.md",
    "introduction/quickstart.md",
    "introduction/concepts.md",
    "introduction/coming-from.md",
    "introduction/cli.md",
    "introduction/viz.md",
    "coordinates/index.md",
    "coordinates/projections.md",
    "coordinates/distortions.md",
    "coordinates/frames.md",
    "coordinates/observed-place.md",
    "sphere/index.md",
    "sphere/healpix.md",
    "sphere/moc.md",
    "sphere/sht.md",
    "sphere/sparse-maps.md",
    "catalogs/index.md",
    "catalogs/crossmatch.md",
    "catalogs/containers.md",
    "maps/index.md",
    "maps/resampling.md",
    "maps/coadds.md",
    "maps/cubes.md",
    "maps/healpix-mapmaking.md",
    "spectra/index.md",
    "spectra/spectral-axes.md",
    "spectra/kinematics.md",
    "spectra/photometry.md",
    "spectra/extinction.md",
    "spectra/cosmology.md",
    "models/index.md",
    "models/fields.md",
    "models/scenes.md",
    "models/likelihoods.md",
    "models/psf-conventions.md",
    "models/encoders.md",
    "models/formulation.md",
    "examples/index.md",
    "examples/gallery.md",
    "project/changelog.md",
    "project/roadmap.md",
    "project/release.md",
    "project/contributing.md",
    "project/authoring.md",
    "reference/index.md",
    "reference/runtime-contract.md",
    "performance/highlights.md",
    "performance/benchmarks.md",
    "performance/parity.md",
    "performance/scaling.md",
)

# Non-published trees that must stay outside docs/.
NOTE_TREES = ("projects", "research", "reviews", "archive", "roadmap", "formulation")


def _zensical_config() -> dict:
    with (ROOT / "zensical.toml").open("rb") as stream:
        return tomllib.load(stream)


def _nav_pages(nav: object) -> list[str]:
    pages: list[str] = []
    if isinstance(nav, list):
        for item in nav:
            pages.extend(_nav_pages(item))
    elif isinstance(nav, dict):
        for value in nav.values():
            pages.extend(_nav_pages(value))
    elif isinstance(nav, str):
        pages.append(nav)
    return pages


def _guide_pages() -> list[Path]:
    """Hand-authored guide pages (excludes generated reference and site copies)."""
    return [
        path
        for path in sorted(DOCS.rglob("*.md"))
        if path.relative_to(DOCS).parts[0] not in {"reference", "published-examples"}
        and not path.name.startswith("_")
    ]


def _markdown_links(text: str) -> list[str]:
    return re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", text)


def test_required_pages_exist() -> None:
    missing = [name for name in REQUIRED_PAGES if not (DOCS / name).is_file()]
    assert not missing, f"Missing published docs pages: {missing}"


def test_every_published_page_is_in_navigation() -> None:
    config = _zensical_config()
    pages = set(_nav_pages(config["project"].get("nav", [])))
    published = {str(path.relative_to(DOCS)) for path in _guide_pages()}
    unlisted = sorted(published - pages)
    assert not unlisted, f"Published pages missing from the nav: {unlisted}"

    missing = sorted(page for page in pages if not (DOCS / page).is_file())
    assert not missing, f"Nav entries without a page: {missing}"


def test_site_configuration_and_assets_exist() -> None:
    config = _zensical_config()
    theme = config["project"]["theme"]
    assert theme["name"] == "material"
    assert (ROOT / theme["custom_dir"]).is_dir()

    for asset in ("assets/torchsky-logo.svg", "assets/torchsky-logo-hero.svg"):
        assert (DOCS / asset).is_file(), asset
    for stylesheet in config["project"].get("extra_css", []):
        assert (DOCS / stylesheet).is_file(), stylesheet
    assert (ROOT / "overrides" / "home.html").is_file()
    assert (ROOT / "overrides" / "main.html").is_file()
    assert (ROOT / "overrides" / "partials" / "actions.html").is_file()


def test_notes_are_outside_the_published_tree() -> None:
    for tree in NOTE_TREES:
        local_tree = ROOT / "notes" / tree
        assert not local_tree.exists() or local_tree.is_dir(), f"notes/{tree}"
        assert not (DOCS / tree).exists(), f"docs/{tree} must not be published"
    assert not (DOCS / "README.md").exists(), "use docs/index.md as the entry point"


def test_readme_points_at_the_published_index() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/index.md" in readme
    assert "docs/README.md" not in readme


def test_relative_doc_links_resolve() -> None:
    broken: list[str] = []
    for page in _guide_pages():
        text = page.read_text(encoding="utf-8")
        for link in _markdown_links(text):
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = link.split("#", 1)[0]
            if not target:
                continue
            if "published-examples/" in target:
                # Generated at docs-build time from examples/ by
                # scripts/sync_docs_examples.sh; check the source script.
                source = ROOT / "examples" / Path(target).name
                if not source.is_file():
                    broken.append(f"{page.relative_to(ROOT)} -> {link}")
                continue
            if not (page.parent / target).resolve().exists():
                broken.append(f"{page.relative_to(ROOT)} -> {link}")
    assert not broken, f"Broken relative doc links: {broken}"


def test_examples_page_lists_existing_scripts() -> None:
    page = (DOCS / "examples" / "index.md").read_text(encoding="utf-8")
    referenced = [
        link
        for link in _markdown_links(page)
        if link.endswith(".py") and not link.startswith(("http://", "https://"))
    ]
    assert referenced, "examples/index.md must link runnable example scripts"
    missing: list[str] = []
    for link in referenced:
        name = Path(link.split("#", 1)[0]).name
        if not (ROOT / "examples" / name).is_file():
            missing.append(link)
    assert not missing, f"examples/index.md links missing scripts: {missing}"
    assert (ROOT / "scripts" / "sync_docs_examples.sh").is_file()


def test_published_examples_are_generated_not_authored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "docs/published-examples/" in gitignore
    script = (ROOT / "scripts" / "sync_docs_examples.sh").read_text(encoding="utf-8")
    assert "examples" in script and "docs/published-examples" in script


def test_docs_do_not_advertise_removed_io_aliases() -> None:
    removed = (
        "torchsky.read(",
        "torchsky.write(",
        "torchsky.get_header(",
        "torchsky.get_wcs(",
    )
    offenders: list[str] = []
    for page in _guide_pages():
        if page.name == "changelog.md":
            # The changelog records removals by name; it must not break this check.
            continue
        text = page.read_text(encoding="utf-8")
        for pattern in removed:
            if pattern in text:
                offenders.append(f"{page.relative_to(DOCS)}: {pattern}")
    assert not offenders, f"Docs reference removed torchsky I/O aliases: {offenders}"


def test_api_manifest_is_valid_json() -> None:
    manifest = json.loads((DOCS / "api-manifest.json").read_text(encoding="utf-8"))
    assert manifest
