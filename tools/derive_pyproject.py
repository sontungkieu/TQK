#!/usr/bin/env python3
"""Derive pyproject.toml from the published Fk-Diffusion-Steering/requirements.txt.

The published reproduction pins an exact pip freeze. This script transcribes that
freeze verbatim so the uv project matches the original environment instead of
re-resolving newer versions.

ImageReward's legacy setup.py (no pyproject.toml) imports pkg_resources, which
setuptools dropped from its default install after 81.x. setup/setup.sh works around
this by installing setuptools==69.5.1 and then installing with --no-build-isolation.
--strategy chooses which uv-native equivalent to emit:

  1  [tool.uv] constraint-dependencies = ["setuptools==69.5.1"]
  2  [tool.uv] build-constraint-dependencies = ["setuptools==69.5.1"]
  3  setuptools==69.5.1 as a project dependency + no-build-isolation-package
  4  drop the git source for image-reward and use the PyPI wheel (fallback only)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "Fk-Diffusion-Steering" / "requirements.txt"
OUT = ROOT / "pyproject.toml"

GIT_RE = re.compile(r"^-e\s+git\+(?P<url>[^@#\s]+?)(?:@(?P<rev>[0-9a-fA-F]{7,40}))?#egg=(?P<egg>[A-Za-z0-9_.-]+)$")

BUILD_SETUPTOOLS = "69.5.1"
PYPI_FALLBACK = "1.5"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=int, default=3, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    dependencies: list[str] = []
    sources: list[tuple[str, str, str]] = []
    for raw in REQ.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = GIT_RE.match(line)
        if match:
            name = match.group("egg").replace("_", "-")
            if name == "image-reward" and args.strategy == 4:
                dependencies.append(f"image-reward=={PYPI_FALLBACK}")
                continue
            dependencies.append(name)
            if not match.group("rev"):
                raise SystemExit(f"git dependency without a full revision: {line}")
            sources.append((name, match.group("url"), match.group("rev")))
            continue
        dependencies.append(line)

    if args.strategy == 3 and f"setuptools=={BUILD_SETUPTOOLS}" not in dependencies:
        dependencies.append(f"setuptools=={BUILD_SETUPTOOLS}")

    tool_uv = ["[tool.uv]", "package = false"]
    if args.strategy == 1:
        tool_uv.append(f'constraint-dependencies = ["setuptools=={BUILD_SETUPTOOLS}"]')
    if args.strategy == 2:
        tool_uv.append(f'build-constraint-dependencies = ["setuptools=={BUILD_SETUPTOOLS}"]')
    if args.strategy == 3:
        tool_uv.append('no-build-isolation-package = ["image-reward"]')

    body = [
        "[project]",
        'name = "tqk"',
        'version = "0.1.0"',
        'description = "Calibrated single-stage pruning for SD1.5 diffusion inference"',
        'readme = "README.md"',
        'requires-python = ">=3.10,<3.11"',
        "dependencies = [",
    ]
    body += [f'    "{dep}",' for dep in dependencies]
    body += ["]", ""]
    body += tool_uv
    if sources:
        body += ["", "# Exact upstream revisions for the editable git requirements.", "[tool.uv.sources]"]
        for name, url, rev in sources:
            body.append(f'{name} = {{ git = "{url}", rev = "{rev}" }}')
    body += [
        "",
        "# ImageReward's legacy setup.py imports pkg_resources; setup.sh pins",
        "# setuptools 69.5.1 for that build, so the isolated build env must too.",
        "[tool.uv.extra-build-dependencies]",
        f'image-reward = ["setuptools=={BUILD_SETUPTOOLS}"]',
        "",
    ]

    OUT.write_text("\n".join(body))
    print(f"wrote {OUT} strategy={args.strategy} deps={len(dependencies)} git_sources={len(sources)}")


if __name__ == "__main__":
    main()
