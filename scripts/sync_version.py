"""Keep the version consistent: pyproject.toml is the source of truth.

    python scripts/sync_version.py --check          # README badge and CHANGELOG agree (CI, tests)
    python scripts/sync_version.py --write          # rewrite the README badge from pyproject
    python scripts/sync_version.py --tag v1.0.0     # release workflow: the tag must match
    python scripts/sync_version.py --notes 1.0.0    # print that version's CHANGELOG section
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent
BADGE = re.compile(r"img\.shields\.io/badge/version-(\d+\.\d+\.\d+)-")


def project_version(root: Path = ROOT) -> str:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def readme_version(root: Path = ROOT) -> str | None:
    match = BADGE.search((root / "README.md").read_text(encoding="utf-8"))
    return match.group(1) if match else None


def changelog_section(version: str, root: Path = ROOT) -> str | None:
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", text, re.M | re.S)
    return match.group(1).strip() if match else None


def problems(root: Path = ROOT) -> list[str]:
    version = project_version(root)
    found = []
    if readme_version(root) != version:
        found.append(f"README badge shows {readme_version(root)}, pyproject says {version}")
    if changelog_section(version, root) is None:
        found.append(f"CHANGELOG has no section for {version}")
    return found


def write_badge(root: Path = ROOT) -> None:
    path = root / "README.md"
    version = project_version(root)
    path.write_text(
        BADGE.sub(f"img.shields.io/badge/version-{version}-", path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--write", action="store_true")
    group.add_argument("--tag")
    group.add_argument("--notes")
    a = p.parse_args()
    if a.write:
        write_badge()
        return 0
    if a.tag:
        expected = f"v{project_version()}"
        if a.tag != expected:
            sys.stderr.write(f"tag {a.tag} does not match pyproject ({expected})\n")
            return 1
        return 0
    if a.notes:
        section = changelog_section(a.notes)
        if section is None:
            sys.stderr.write(f"no CHANGELOG section for {a.notes}\n")
            return 1
        sys.stdout.write(section + "\n")
        return 0
    issues = problems()
    for issue in issues:
        sys.stderr.write(issue + "\n")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
