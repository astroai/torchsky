#!/usr/bin/env bash
# Publish example scripts with the documentation site.
#
# The site must not link outside its own tree, so every runnable example in
# examples/ is copied under docs/published-examples/ before the Zensical build.
# The copy is generated (git-ignored) and never edited by hand.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/docs/published-examples"
rm -rf "${DEST}"
mkdir -p "${DEST}"
shopt -s nullglob
for f in "${ROOT}/examples"/*.py; do
  cp "$f" "${DEST}/"
done
# Generated index: hide the Edit/View actions, which would point at raw/main
# files that do not exist.
printf -- '---\nhide:\n  - edit\n  - view\n---\n\n# Published examples\n\nCopied from `examples/` at docs-build time. Do not edit.\n' \
  > "${DEST}/README.md"
