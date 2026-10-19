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
    "install.md",
    "quickstart.md",
    "getting-started.md",
    "compatibility.md",
    "wcs.md",
    "coordinates.md",
    "sphere.md",
    "catalogs.md",
    "spectral.md",
    "map.md",
    "scene.md",
    "field.md",
    "encoders.md",
    "ml.md",
    "io.md",
    "table.md",
    "cli.md",
    "viz.md",
    "native-likelihood.md",
    "implicit_forward.md",
    "psf-response-conventions.md",
    "astrometric_calibration.md",
    "photometry-encoders.md",
    "catalogue-foundations.md",
    "api.md",
    "architecture.md",
    "benchmarks.md",
    "examples.md",
    "changelog.md",
    "roadmap.md",
    "release.md",
    "contributing.md",
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


def _markdown_links(text: str) -> list[str]:
    return re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", text)


def test_required_pages_exist() -> None:
    missing = [name for name in REQUIRED_PAGES if not (DOCS / name).is_file()]
    assert not missing, f"Missing published docs pages: {missing}"


def test_every_published_page_is_in_navigation() -> None:
    config = _zensical_config()
    pages = set(_nav_pages(config["project"].get("nav", [])))
    published = {path.name for path in DOCS.glob("*.md")}
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


def test_notes_are_outside_the_published_tree() -> None:
    for tree in NOTE_TREES:
        assert (ROOT / "notes" / tree).is_dir(), f"notes/{tree}"
        assert not (DOCS / tree).exists(), f"docs/{tree} must not be published"
    assert not (DOCS / "README.md").exists(), "use docs/index.md as the entry point"


def test_readme_points_at_the_published_index() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/index.md" in readme
    assert "docs/README.md" not in readme


def test_relative_doc_links_resolve() -> None:
    broken: list[str] = []
    for page in DOCS.glob("*.md"):
        text = page.read_text(encoding="utf-8")
        for link in _markdown_links(text):
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = link.split("#", 1)[0]
            if not target:
                continue
            if not (page.parent / target).resolve().exists():
                broken.append(f"{page.relative_to(ROOT)} -> {link}")
    assert not broken, f"Broken relative doc links: {broken}"


def test_examples_page_lists_existing_scripts() -> None:
    page = (DOCS / "examples.md").read_text(encoding="utf-8")
    referenced = [
        link
        for link in _markdown_links(page)
        if link.endswith(".py") and not link.startswith(("http://", "https://"))
    ]
    assert referenced, "examples.md must link runnable example scripts"
    missing = [
        link
        for link in referenced
        if not (DOCS / link.split("#", 1)[0]).resolve().is_file()
    ]
    assert not missing, f"examples.md links missing scripts: {missing}"


def test_docs_do_not_advertise_removed_io_aliases() -> None:
    removed = (
        "torchsky.read(",
        "torchsky.write(",
        "torchsky.get_header(",
        "torchsky.get_wcs(",
    )
    offenders: list[str] = []
    for page in DOCS.glob("*.md"):
        if page.name == "changelog.md":
            # The changelog records removals by name; it must not break this check.
            continue
        text = page.read_text(encoding="utf-8")
        for pattern in removed:
            if pattern in text:
                offenders.append(f"{page.name}: {pattern}")
    assert not offenders, f"Docs reference removed torchsky I/O aliases: {offenders}"


def test_api_manifest_is_valid_json() -> None:
    manifest = json.loads((DOCS / "api-manifest.json").read_text(encoding="utf-8"))
    assert manifest
