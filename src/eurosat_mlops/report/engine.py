"""FPDF2 report engine: EFREI colours, Epilogue for text, Overpass Mono for code and figures.

Layout rules enforced here and measured afterwards by scripts/check_report.py:
- a figure that does not fit the rest of the page is first scaled down (ratio kept, down to
  62 %), otherwise deferred: the text that follows flows into the gap and the figure goes to
  the top of the next page;
- a heading never sits alone at the bottom of a page.
"""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import FontFace

NAVY = (22, 55, 103)
ROSE = (255, 67, 184)
DARK = (5, 24, 50)
BLUE = (12, 120, 180)
GREY = (108, 117, 125)
LIGHT = (240, 243, 248)
WHITE = (255, 255, 255)

MARGIN = 17
MIN_FIG_RATIO = 0.62
HEADING_KEEP_MM = 40
ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "reports" / "assets"
FONTS = ASSETS / "fonts"

_RICH = re.compile(r"(\*\*.+?\*\*|`.+?`)")
_NUMERIC = re.compile(r"^[\d\s.,x%+\-/<>=()]+$")


class Report(FPDF):
    """A4 report with a running header, a numbered footer and flow-aware figures."""

    def __init__(self, title: str, short_title: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.short_title = short_title
        self.set_margins(MARGIN, 20, MARGIN)
        self.set_auto_page_break(True, margin=18)
        self.add_font("Body", "", str(FONTS / "Epilogue-Regular.ttf"))
        self.add_font("Body", "B", str(FONTS / "Epilogue-Bold.ttf"))
        self.add_font("Body", "I", str(FONTS / "Epilogue-Italic.ttf"))
        self.add_font("Display", "", str(FONTS / "Epilogue-ExtraBold.ttf"))
        self.add_font("Mono", "", str(FONTS / "OverpassMono-Regular.ttf"))
        self.add_font("Mono", "B", str(FONTS / "OverpassMono-Bold.ttf"))
        self.set_title(title)
        self.set_author("Adam Beloucif, Emilien Morice")
        self.set_creator("EFREI Paris, Mastère Data Engineering et IA")
        self.pending: list[tuple[str, str, float]] = []
        self._after_h1 = False
        self.fig_no = 0
        self.tab_no = 0

    # ----------------------------------------------------------------- frame
    def header(self) -> None:
        if self.page_no() == 1:  # the cover carries its own layout
            return
        self.set_fill_color(*NAVY)
        self.rect(0, 0, self.w, 3, "F")
        self.set_xy(MARGIN, 7)
        self.set_font("Body", "", 7.5)
        self.set_text_color(*GREY)
        self.cell(0, 4, self.short_title, align="L")
        logo = ASSETS / "efrei-logo-couleur.png"
        if logo.exists():
            self.image(str(logo), x=self.w - MARGIN - 16, y=5.5, h=7)  # height only: no distortion
        self.set_y(20)

    def footer(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_draw_color(*ROSE)
        self.set_line_width(0.4)
        self.line(MARGIN, self.y, MARGIN + 12, self.y)
        self.set_font("Body", "", 7.5)
        self.set_text_color(*GREY)
        self.cell(0, 6, f"{self.page_no()}", align="R")

    def space_left(self) -> float:
        return self.h - self.b_margin - self.get_y()

    # ------------------------------------------------------------ rich text
    def _style(self, kind: str, size: float) -> None:
        if kind == "bold":
            self.set_font("Body", "B", size)
        elif kind == "code":
            self.set_font("Mono", "", size * 0.86)
        else:
            self.set_font("Body", "", size)

    def rich(self, text: str, size: float, line: float) -> None:
        """Write text with **bold** and `code` spans, wrapping on whole words only."""
        words: list[list[tuple[str, str]]] = [[]]
        for part in _RICH.split(text):
            if not part:
                continue
            kind = "bold" if part.startswith("**") else "code" if part.startswith("`") else "plain"
            if kind != "plain":
                part = part[2:-2] if kind == "bold" else part[1:-1]
            for token in re.findall(r"\S+|\s+", part):
                if token.isspace():
                    words.append([])
                else:
                    words[-1].append((kind, token))
        self._style("plain", size)
        space = self.get_string_width(" ")
        right = self.w - self.r_margin
        for word in filter(None, words):
            width = 0.0
            for kind, token in word:
                self._style(kind, size)
                width += self.get_string_width(token)
            at_start = self.get_x() <= self.l_margin + 0.01
            if not at_start and self.get_x() + space + width > right:
                self.ln(line)
                at_start = True
            if not at_start:
                self.set_x(self.get_x() + space)
            for kind, token in word:
                self._style(kind, size)
                self.cell(self.get_string_width(token), line, token)
        self._style("plain", size)

    # -------------------------------------------------------------- blocks
    def _flush_pending(self) -> None:
        self._after_h1 = False
        while self.pending:
            path, caption, width = self.pending[0]
            if not self._place_figure(path, caption, width, allow_break=False):
                return
            self.pending.pop(0)

    def h1(self, text: str) -> None:
        self._flush_pending()
        if self.space_left() < HEADING_KEEP_MM:
            self.add_page()
            self._flush_pending()
        self.ln(3)
        self.start_section(text, level=0)
        self.set_font("Display", "", 15)
        self.set_text_color(*NAVY)
        self.multi_cell(0, 7.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.set_fill_color(*ROSE)
        self.rect(MARGIN, self.get_y() + 0.5, 14, 0.9, "F")
        self.ln(3.5)
        self._after_h1 = True

    def h2(self, text: str) -> None:
        directly_under_h1 = self._after_h1
        self._flush_pending()
        if not directly_under_h1 and self.space_left() < HEADING_KEEP_MM - 10:
            self.add_page()
            self._flush_pending()
        self.ln(1.5)
        self.start_section(text, level=1)
        self.set_font("Body", "B", 10.5)
        self.set_text_color(*BLUE)
        self.multi_cell(0, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.ln(0.8)

    def para(self, text: str, size: float = 10) -> None:
        self._flush_pending()
        self.set_text_color(*DARK)
        self.set_x(MARGIN)
        self.rich(text, size, size * 0.5)
        self.ln(size * 0.5 + 1.6)

    def bullets(self, items: list[str], size: float = 10) -> None:
        self._flush_pending()
        for item in items:
            self.set_x(MARGIN + 2)
            self.set_font("Body", "B", size)
            self.set_text_color(*ROSE)
            self.cell(4, size * 0.5, "-")
            self.set_text_color(*DARK)
            self.set_left_margin(MARGIN + 6)
            self.set_x(MARGIN + 6)
            self.rich(item, size, size * 0.5)
            self.set_left_margin(MARGIN)
            self.ln(size * 0.5 + 0.8)
        self.ln(1)

    def data_table(
        self,
        rows: list[list[str]],
        widths: list[float],
        caption: str,
        size: float = 8.2,
        align: str | None = None,
        bold_row: int | None = None,
    ) -> None:
        """rows[0] is the header. align: one letter per column (L, C, R)."""
        self._flush_pending()
        self.tab_no += 1
        self.set_font("Body", "", size)
        self.set_text_color(*DARK)
        self.set_draw_color(210, 217, 228)
        self.set_line_width(0.15)
        head = FontFace(emphasis="BOLD", color=WHITE, fill_color=NAVY)
        self.set_fill_color(*WHITE)
        with (
            self.unbreakable() as doc,
            doc.table(
                col_widths=widths,
                width=sum(widths),
                line_height=size * 0.5,
                headings_style=head,
                text_align=tuple(align) if align else "LEFT",
                padding=(1, 1.4),
                first_row_as_headings=True,
            ) as tab,
        ):
            tab.row([str(c) for c in rows[0]])
            for number, cells in enumerate(rows[1:], start=1):
                row = tab.row()
                fill = LIGHT if number % 2 == 0 else WHITE
                for cell in cells:
                    text = str(cell)
                    family = "Mono" if _NUMERIC.match(text) else "Body"
                    emphasis = "B" if number == bold_row else ""
                    row.cell(
                        text,
                        style=FontFace(family=family, emphasis=emphasis, fill_color=fill),
                    )
        self._caption(f"Tableau {self.tab_no}. {caption}")

    def _caption(self, text: str) -> None:
        self.ln(1)
        self.set_x(MARGIN)
        self.set_font("Body", "I", 8)
        self.set_text_color(*GREY)
        self.multi_cell(0, 4, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.ln(2.5)

    def figure(self, path: Path, caption: str, width: float | None = None) -> int:
        """Place a figure now if it fits (possibly scaled), else defer it. Returns its number."""
        self.fig_no += 1
        full = f"Figure {self.fig_no}. {caption}"
        width = width or (self.w - 2 * MARGIN)
        if self.pending or not self._place_figure(str(path), full, width, allow_break=False):
            self.pending.append((str(path), full, width))
        return self.fig_no

    def _place_figure(self, path: str, caption: str, width: float, allow_break: bool) -> bool:
        from PIL import Image

        with Image.open(path) as im:
            ratio = im.height / im.width
        caption_h = 9
        avail = self.space_left() - caption_h - 2
        height = width * ratio
        body = self.h - 20 - self.b_margin - caption_h - 2
        if height > body:  # taller than a whole page: fit the page body, ratio kept
            width, height = body / ratio, body
        if height > avail:
            if avail >= height * MIN_FIG_RATIO:
                width, height = avail / ratio, avail
            elif not allow_break:
                return False
            else:
                self.add_page()
        x = (self.w - width) / 2
        self.image(path, x=x, y=self.get_y(), w=width, h=height)
        self.set_y(self.get_y() + height)
        self._caption(caption)
        return True

    def finish(self) -> None:
        while self.pending:
            path, caption, width = self.pending.pop(0)
            self._place_figure(path, caption, width, allow_break=True)
