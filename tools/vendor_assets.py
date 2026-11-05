"""Fetch the UI icons and fonts from npm into ui/ so the delivered app uses no CDN.

    python tools/vendor_assets.py

Needs node and npm. Sources: reicon (icons, MIT), @fontsource-variable/epilogue and
@fontsource/overpass-mono (fonts, SIL OFL). The result is committed, this script only
documents where it came from and makes it reproducible.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

UI = Path(__file__).resolve().parent.parent / "ui"
ICONS = ("Upload", "AlertTriangle", "CheckCircle")
FONTS = {
    "@fontsource-variable/epilogue": ["epilogue-latin-wght-normal.woff2"],
    "@fontsource/overpass-mono": [
        "overpass-mono-latin-400-normal.woff2",
        "overpass-mono-latin-600-normal.woff2",
    ],
}


def npm_install(workdir: Path) -> Path:
    npm = shutil.which("npm")
    if npm is None:
        sys.exit("npm not found")
    subprocess.run([npm, "init", "-y"], cwd=workdir, check=True, capture_output=True)  # noqa: S603
    subprocess.run(  # noqa: S603
        [npm, "install", "--silent", "reicon", *FONTS], cwd=workdir, check=True
    )
    return workdir / "node_modules"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        modules = npm_install(Path(tmp))
        icons_out = UI / "vendor" / "reicon"
        (icons_out / "icons").mkdir(parents=True, exist_ok=True)
        shutil.copy(modules / "reicon" / "createIcon.js", icons_out / "createIcon.js")
        for name in ICONS:
            shutil.copy(modules / "reicon" / "icons" / f"{name}.js", icons_out / "icons")
        (UI / "fonts").mkdir(exist_ok=True)
        for package, files in FONTS.items():
            for f in files:
                shutil.copy(modules / package / "files" / f, UI / "fonts" / f)
    print(f"vendored {len(ICONS)} icons and {sum(map(len, FONTS.values()))} font files into {UI}")


if __name__ == "__main__":
    main()
