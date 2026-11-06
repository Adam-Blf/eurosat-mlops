"""Quality gate for the generated PDF report.

Fails (exit 1) when:
- the page count is outside the expected range;
- a page other than the last leaves more than 40 mm empty at the bottom or between two blocks;
- text crosses the side margins;
- an embedded font is not Epilogue or Overpass Mono;
- a forbidden character (em dash, en dash, middle dot) or word appears in the extracted text.

Usage: python scripts/check_report.py [reports/rapport-eurosat-mlops.pdf]
Needs PyMuPDF (installed by the `report` extra).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pymupdf as fitz

MM = 72 / 25.4
MAX_BLANK_MM = 40
MARGIN_MM = 15
HEADER_MM = 18
FOOTER_MM = 18
PAGES = range(10, 15)
ALLOWED_FONTS = ("Epilogue", "OverpassMono")
# Code points rather than characters, so this file passes its own typography rule.
FORBIDDEN_CHARS = {chr(0x2014): "em dash", chr(0x2013): "en dash", chr(0x00B7): "middle dot"}
FORBIDDEN_WORDS = ("Claude", "Anthropic", "ChatGPT", "Copilot")
# Layout fillers pass every geometric check: the sister project shipped a cover with a
# test summary and a one-word conclusion while its checker was green. Refuse them.
PLACEHOLDERS = re.compile(
    r"texte de test|paragraphe de test|lorem ipsum|\bTODO\b|\bTBD\b|^\s*Test\.\s*$",
    re.I | re.M,
)
DEFAULT_PDF = Path("reports/rapport-eurosat-mlops.pdf")


def content_boxes(page: fitz.Page) -> list[fitz.Rect]:
    boxes = [fitz.Rect(b[:4]) for b in page.get_text("blocks") if b[4].strip()]
    boxes += [page.get_image_bbox(img) for img in page.get_images(full=True)]
    boxes += [fitz.Rect(d["rect"]) for d in page.get_drawings() if d["rect"].height > 3 * MM]
    top, bottom = HEADER_MM * MM, page.rect.height - FOOTER_MM * MM
    return sorted((r for r in boxes if r.y0 > top and r.y1 < bottom), key=lambda r: r.y0)


def blanks(page: fitz.Page, first_page: bool) -> list[tuple[str, float]]:
    """Empty stretches taller than MAX_BLANK_MM, between blocks or at the page bottom."""
    found, bottom = [], HEADER_MM * MM
    for box in content_boxes(page):
        if box.y0 - bottom > MAX_BLANK_MM * MM and not first_page:
            found.append(("gap", (box.y0 - bottom) / MM))
        bottom = max(bottom, box.y1)
    free = (page.rect.height - FOOTER_MM * MM - bottom) / MM
    if free > MAX_BLANK_MM:
        found.append(("page bottom", free))
    return found


def font_errors(doc: fitz.Document) -> list[str]:
    seen = {font[3].split("+")[-1] for page in doc for font in page.get_fonts()}
    return [
        f"font {name!r} is not Epilogue or Overpass Mono"
        for name in sorted(seen)
        if not name.startswith(ALLOWED_FONTS)
    ]


def check(pdf: Path) -> list[str]:
    doc = fitz.open(pdf)
    errors = []
    if doc.page_count not in PAGES:
        errors.append(f"{doc.page_count} pages, expected {PAGES.start} to {PAGES.stop - 1}")
    errors += font_errors(doc)
    for number, page in enumerate(doc, start=1):
        text = page.get_text()
        errors += [f"page {number}: {name}" for ch, name in FORBIDDEN_CHARS.items() if ch in text]
        errors += [
            f"page {number}: placeholder {m.group(0).strip()!r}"
            for m in PLACEHOLDERS.finditer(text)
        ]
        errors += [
            f"page {number}: forbidden word {w!r}"
            for w in FORBIDDEN_WORDS
            if re.search(rf"\b{w}\b", text, re.I)
        ]
        for block in page.get_text("blocks"):
            if (
                block[6] == 0
                and block[4].strip()
                and (block[0] < MARGIN_MM * MM or block[2] > page.rect.width - MARGIN_MM * MM)
            ):
                errors.append(f"page {number}: text outside margin, {block[4].strip()[:40]!r}")
        if number < doc.page_count:
            errors += [
                f"page {number}: {size:.0f} mm empty ({kind})"
                for kind, size in blanks(page, first_page=number == 1)
            ]
    print(f"{pdf.name}: {doc.page_count} pages")
    return errors


def main() -> int:
    pdf = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF)
    errors = check(pdf)
    for error in errors:
        print("  FAIL", error)
    print("OK" if not errors else f"{len(errors)} defect(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
