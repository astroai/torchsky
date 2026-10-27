#!/usr/bin/env python3
"""Audit public docstrings and enforce a shrinking coverage baseline.

Every public name must have a summary line. Public callables should also
document parameters, returns, and raises; Tier-1 names additionally need a
runnable example. Existing gaps are recorded in
``docs/data/docstring-baseline.json``. The baseline may shrink, never grow, so a
new public API cannot land without documentation while legacy debt is burned
down deliberately.

    pixi run docs-docstrings              # report
    pixi run docs-docstrings-baseline     # rewrite the baseline (review the diff!)
    pixi run docs-docstrings-check        # fail on new debt or a Tier-1 gap
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MANIFEST = ROOT / "docs" / "api-manifest.json"
TIER1 = ROOT / "docs" / "tier1.txt"
BASELINE = ROOT / "docs" / "data" / "docstring-baseline.json"
REGISTRY = ROOT / "docs" / "reference" / "_registry.toml"
SECTION = re.compile(r"^\s*(Args|Arguments|Parameters|Returns|Yields|Raises|Example|Examples):\s*$", re.M)


def _requires_returns(obj: object) -> bool:
    try:
        annotation = inspect.signature(obj).return_annotation
    except (TypeError, ValueError):
        return False
    return annotation not in {inspect.Signature.empty, None, "None"}


def _required_sections(obj: object) -> set[str]:
    required = {"summary"}
    if inspect.isclass(obj) or callable(obj):
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and signature.parameters:
            required.add("Args")
        if inspect.isfunction(obj) and _requires_returns(obj):
            required.add("Returns")
    return required


def _present_sections(doc: str) -> set[str]:
    present = set()
    if doc.strip():
        present.add("summary")
    for match in SECTION.finditer(doc):
        name = match.group(1)
        present.add("Args" if name in {"Args", "Arguments", "Parameters"} else name)
        present.add("Returns" if name in {"Returns", "Yields"} else name)
        present.add("Examples" if name in {"Example", "Examples"} else name)
    return present


def _registered_type_alias_docs() -> dict[str, str]:
    registry = tomllib.loads(REGISTRY.read_text(encoding="utf-8"))
    docs = registry.get("public_type_aliases", {})
    if not isinstance(docs, dict):
        raise TypeError("public_type_aliases must be a TOML table")
    for name, summary in docs.items():
        if not isinstance(summary, str) or not summary.strip() or "\n" in summary:
            raise ValueError(f"public type-alias summary must be one non-empty line: {name}")
    return docs


def collect() -> dict[str, list[str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    alias_docs = _registered_type_alias_docs()
    public_names = {
        f"{namespace}.{name}" for namespace, names in manifest.items() for name in names
    }
    stale_alias_docs = sorted(set(alias_docs) - public_names)
    if stale_alias_docs:
        raise ValueError(f"registered type aliases are not public: {stale_alias_docs}")
    tier1 = {
        line.strip()
        for line in TIER1.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    debt: dict[str, list[str]] = {}
    for namespace, names in manifest.items():
        module = importlib.import_module(namespace)
        for name in names:
            obj = getattr(module, name)
            if inspect.ismodule(obj):
                continue
            full = f"{namespace}.{name}"
            doc = inspect.getdoc(obj) or ""
            if full in alias_docs:
                if getattr(obj, "__module__", None) != "typing":
                    raise TypeError(f"registered public type alias is not a typing object: {full}")
                doc = alias_docs[full]
            present = _present_sections(doc)
            if full in tier1:
                missing = sorted(_required_sections(obj) - present)
                if "Examples" not in present:
                    missing.append("Examples")
            else:
                # Legacy names are ratcheted on the summary line only; section
                # completeness for Tier-1 is what the baseline burns down.
                missing = [] if "summary" in present else ["summary"]
            if missing:
                debt[full] = sorted(missing)
    return dict(sorted(debt.items()))


def write_baseline(debt: dict[str, list[str]]) -> None:
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(debt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check(debt: dict[str, list[str]]) -> int:
    if not BASELINE.is_file():
        print(f"missing {BASELINE.relative_to(ROOT)}; run --write-baseline once")
        return 1
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    added = sorted(set(debt) - set(baseline))
    grown = sorted(
        name
        for name in set(debt) & set(baseline)
        if not set(debt[name]) <= set(baseline[name])
    )
    improved = sorted(set(baseline) - set(debt))
    if added or grown:
        print("[FAIL] docstring debt grew:")
        for name in added:
            print(f"  new: {name} missing {debt[name]}")
        for name in grown:
            print(f"  worse: {name} now missing {sorted(set(debt[name]) - set(baseline[name]))}")
        print("Document the new API or explain why it belongs in the baseline.")
        return 1
    print(
        f"[OK] docstring ratchet holds ({len(debt)} names with debt, "
        f"{len(improved)} improved since the baseline)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-baseline", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    debt = collect()
    if args.write_baseline:
        write_baseline(debt)
        print(f"wrote {BASELINE.relative_to(ROOT)} ({len(debt)} names with debt)")
        return 0
    if args.check:
        return check(debt)
    for name, missing in debt.items():
        print(f"{name}: {', '.join(missing)}")
    print(f"\n{len(debt)} public names have docstring debt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
