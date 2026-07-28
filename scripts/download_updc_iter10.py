#!/usr/bin/env python3

from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urljoin
from urllib.request import Request, urlopen


PAGE_URL = "https://nadc.china-vo.org/res/r101436/?lang=en"

OUTPUT_DIR = Path(
    "/arc/projects/ots/jwst_smacs0723/data"
)

BANDS = (
    "F090W",
    "F150W",
    "F200W",
    "F277W",
    "F356W",
    "F444W",
)

FILENAMES = {
    band: f"{band}b_UPDC-v2_iter10_i2d.fits.fz"
    for band in BANDS
}

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 Chrome/130 Safari/537.36"
)


def fetch_page(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urlopen(request, timeout=120) as response:
        raw = response.read()

    return html.unescape(
        raw.decode("utf-8", errors="replace")
    )


def candidate_urls(page: str, filename: str) -> list[str]:
    """Find likely download URLs near occurrences of filename."""

    decoded_page = unquote(page)
    positions = [
        match.start()
        for match in re.finditer(
            re.escape(filename),
            decoded_page,
            flags=re.IGNORECASE,
        )
    ]

    if not positions:
        raise RuntimeError(
            f"{filename!r} was not found in the resource page"
        )

    candidates: list[tuple[int, int, str]] = []

    attribute_pattern = re.compile(
        r"""(?:
            href |
            data-href |
            data-url |
            download-url |
            onclick
        )\s*=\s*["']([^"']+)["']""",
        flags=re.IGNORECASE | re.VERBOSE,
    )

    absolute_pattern = re.compile(
        r"""https?://[^\s"'<>\\]+""",
        flags=re.IGNORECASE,
    )

    for position in positions:
        start = max(0, position - 6000)
        stop = min(len(decoded_page), position + 6000)
        window = decoded_page[start:stop]

        values = [
            match.group(1)
            for match in attribute_pattern.finditer(window)
        ]
        values.extend(
            match.group(0)
            for match in absolute_pattern.finditer(window)
        )

        for raw_value in values:
            value = html.unescape(unquote(raw_value.strip()))

            # Extract a URL embedded in an onclick expression.
            embedded = re.search(
                r"""(https?://[^"'()\s]+|/[^"'()\s]+)""",
                value,
            )
            if embedded:
                value = embedded.group(1)

            if not value or value.startswith(
                ("javascript:", "#", "mailto:")
            ):
                continue

            url = urljoin(PAGE_URL, value)
            lower = url.lower()

            score = 0
            if filename.lower() in lower:
                score += 100
            if "download" in lower:
                score += 30
            if "file" in lower:
                score += 10
            if "r101436" in lower or "101436" in lower:
                score += 5

            distance = abs(
                decoded_page.find(raw_value, start, stop)
                - position
            )
            candidates.append((-score, distance, url))

    ordered: list[str] = []
    seen: set[str] = set()

    for _, _, url in sorted(candidates):
        if url in seen:
            continue
        seen.add(url)
        ordered.append(url)

    return ordered


def looks_like_fits(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024 * 1024:
        return False

    with path.open("rb") as handle:
        beginning = handle.read(80)

    return (
        beginning.startswith(b"SIMPLE")
        or beginning.startswith(b"XTENSION")
    )


def download_with_curl(
    url: str,
    destination: Path,
) -> bool:
    command = [
        "curl",
        "--location",
        "--fail",
        "--show-error",
        "--retry", "5",
        "--retry-delay", "5",
        "--connect-timeout", "60",
        "--max-time", "0",
        "--continue-at", "-",
        "--user-agent", USER_AGENT,
        "--referer", PAGE_URL,
        "--output", str(destination),
        url,
    ]

    print("trying:", url, flush=True)

    result = subprocess.run(command, check=False)

    if result.returncode != 0:
        return False

    if not looks_like_fits(destination):
        print(
            f"downloaded content is not a valid-looking FITS: "
            f"{destination}",
            file=sys.stderr,
        )
        destination.unlink(missing_ok=True)
        return False

    return True


def main() -> int:
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required but was not found")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("resource page:", PAGE_URL)
    print("output directory:", OUTPUT_DIR)

    page = fetch_page(PAGE_URL)

    failures: list[str] = []

    for band, filename in FILENAMES.items():
        destination = OUTPUT_DIR / filename

        print(f"\n=== {band} ===")
        print("target:", destination)

        if looks_like_fits(destination):
            print(
                f"already complete: "
                f"{destination.stat().st_size / 1024**2:.2f} MiB"
            )
            continue

        urls = candidate_urls(page, filename)

        success = False
        for url in urls:
            if download_with_curl(url, destination):
                success = True
                break

        if not success:
            failures.append(filename)
            print(
                f"FAILED to resolve/download {filename}",
                file=sys.stderr,
            )
            continue

        print(
            f"finished: {destination.name} "
            f"({destination.stat().st_size / 1024**2:.2f} MiB)"
        )

    print("\n=== downloaded iter10 files ===")
    for band, filename in FILENAMES.items():
        path = OUTPUT_DIR / filename
        if path.exists():
            print(
                f"{band}: {path} "
                f"({path.stat().st_size / 1024**2:.2f} MiB)"
            )

    if failures:
        print("\nfailed files:", file=sys.stderr)
        for filename in failures:
            print("  ", filename, file=sys.stderr)
        return 1

    print("\nAll six UPDC iter10 products downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
