"""
reports/build_pdf_final.py
==========================
Self-contained PDF builder for the HealthRisk AI research article.

    python reports/build_pdf_final.py

Output: reports/HealthRisk_AI_Research_Article.pdf
"""
from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent
ARTICLE = BASE / "healthrisk_ai_research_article.md"
FIGURES = BASE / "figures"
OUT     = BASE / "HealthRisk_AI_Research_Article.pdf"

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY  = (26,  73, 138)
BLUE  = (26, 115, 232)
GREY  = (90,  90,  90)
LGREY = (245, 245, 245)
WHITE = (255, 255, 255)
BLACK = (30,  30,  30)
GREEN = (52,  168,  83)
RED   = (234,  67,  53)

# ── Figure map ────────────────────────────────────────────────────────────────
FIG_MAP = {
    "Figure 1":  "fig1_roc_curves.png",
    "Figure 2":  "fig2_pr_curves.png",
    "Figure 3":  "fig3_calibration.png",
    "Figure 4":  "fig4_shap_importance.png",
    "Figure 5":  "fig5_shap_waterfall.png",
    "Figure 6":  "fig6_pdp.png",
    "Figure 7":  "fig7_survival.png",
    "Figure 8":  "fig8_actuarial.png",
    "Figure 9":  "fig9_credit_risk.png",
    "Figure 10": "fig10_pharma.png",
    "Figure 11": "fig11_simulation.png",
}

# ── Unicode → ASCII map (latin-1 only fonts) ──────────────────────────────────
_UNICODE_MAP: dict[str, str] = {
    "\u2014": "--",     # em dash
    "\u2013": "-",      # en dash
    "\u2212": "-",      # minus sign
    "\u2248": "~=",     # approximately equal
    "\u2192": "->",     # right arrow
    "\u2190": "<-",     # left arrow
    "\u2191": "^",      # up arrow
    "\u2193": "v",      # down arrow
    "\u2713": "[OK]",   # check mark
    "\u2714": "[OK]",   # heavy check
    "\u2718": "[X]",    # cross mark
    "\u2080": "0",      # subscript 0
    "\u2081": "1",      # subscript 1
    "\u2082": "2",      # subscript 2
    "\u00b2": "2",      # superscript 2
    "\u00b3": "3",      # superscript 3
    "\u00b0": " deg",   # degree
    "\u2265": ">=",     # >=
    "\u2264": "<=",     # <=
    "\u00d7": "x",      # multiplication
    "\u00b1": "+/-",    # plus-minus
    "\u03b1": "alpha",
    "\u03b2": "beta",
    "\u03bb": "lambda",
    "\u03c3": "sigma",
    "\u03c7": "chi",
    "\u2019": "'",      # right single quote
    "\u2018": "'",      # left single quote
    "\u201c": '"',      # left double quote
    "\u201d": '"',      # right double quote
    "\u00e9": "e",
    "\u00e8": "e",
    "\u00ea": "e",
    "\u00e0": "a",
    "\u00e2": "a",
    "\u00e9": "e",
    "\u00fc": "u",
    "\u00f6": "o",
    "\u00e4": "a",
    "\u2022": "-",      # bullet
    "\u00a0": " ",      # non-breaking space
    "\u00ad": "-",      # soft hyphen
}


def clean(text: str) -> str:
    """Replace all non-latin-1 characters with ASCII equivalents."""
    for char, repl in _UNICODE_MAP.items():
        text = text.replace(char, repl)
    # Final fallback: replace anything remaining that can't encode as latin-1
    return text.encode("latin-1", errors="replace").decode("latin-1")


def strip_md(text: str) -> str:
    """Strip common markdown formatting for plain-text rendering."""
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)   # bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)         # italic
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # links
    text = text.replace('`', '')
    text = text.replace('>', '')  # blockquote marker
    return clean(text.strip())


# ── PDF class ─────────────────────────────────────────────────────────────────

class ArticlePDF(FPDF):

    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(left=20, top=20, right=20)
        self._table_row: int = 0

    # Header / Footer
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6,
                  "HealthRisk AI -- Bridging Clinical Intelligence and Financial Risk",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.set_draw_color(*BLUE)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, f"Page {self.page_no()}", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Title page
    def title_page(self) -> None:
        self.add_page()

        # Navy banner
        self.set_fill_color(*NAVY)
        self.rect(0, 0, self.w, 52, "F")

        self.set_y(12)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*WHITE)
        self.multi_cell(0, 10,
            "HealthRisk AI\nBridging Clinical Intelligence\nand Financial Risk",
            align="C")

        # Blue subtitle bar
        self.set_fill_color(*BLUE)
        self.rect(0, 52, self.w, 10, "F")
        self.set_y(54)
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(*WHITE)
        self.cell(0, 6, "A Multi-Model Empirical Study", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Author block
        self.set_y(75)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*NAVY)
        self.cell(0, 8, "Vincent Langat Kipkemoi", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*GREY)
        self.cell(0, 6, "HealthRisk Capital Partners  |  July 2026",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.ln(6)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.5)
        self.line(40, self.get_y(), self.w - 40, self.get_y())
        self.ln(8)

        self._kpi_box()

        self.set_y(-30)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 5,
            "Reproduce figures: python reports/generate_figures.py  |  "
            "Figures saved to reports/figures/",
            align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def _kpi_box(self) -> None:
        x0, y0 = self.l_margin, self.get_y()
        bw = self.w - self.l_margin - self.r_margin

        self.set_fill_color(*LGREY)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.4)
        self.rect(x0, y0, bw, 62, "FD")

        self.set_xy(x0 + 4, y0 + 4)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*NAVY)
        self.cell(bw - 8, 7, "Key Results at a Glance",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        kpis = [
            ("Stacking Ensemble AUROC",     "0.831", "target > 0.78  [PASS]", GREEN),
            ("Cost Prediction R2",          "0.28",  "+115% vs CMS-HCC baseline", BLUE),
            ("Cost Prediction MAPE",        "52%",   "target < 52%  [PASS]",    GREEN),
            ("Hospital Default AUROC",      "0.851", "Gini 0.702 [PASS]",       GREEN),
            ("Survival C-index (Ensemble)", "0.762", "target > 0.70  [PASS]",   GREEN),
            ("Actuarial Predictive Ratio",  "0.99",  "target 0.95-1.05  [PASS]",GREEN),
        ]

        col_w = bw / 3 - 2
        for i, (label, value, note, color) in enumerate(kpis):
            row, col = i // 3, i % 3
            cx = x0 + 2 + col * (col_w + 2)
            cy = y0 + 14 + row * 22

            self.set_xy(cx, cy)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*color)
            self.cell(col_w, 8, value, align="C",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            self.set_xy(cx, cy + 8)
            self.set_font("Helvetica", "B", 7)
            self.set_text_color(*NAVY)
            self.cell(col_w, 4, label, align="C",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            self.set_xy(cx, cy + 12)
            self.set_font("Helvetica", "I", 6.5)
            self.set_text_color(*GREY)
            self.cell(col_w, 4, note, align="C",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_y(y0 + 66)


# ── Rendering helpers ─────────────────────────────────────────────────────────

def h2(pdf: ArticlePDF, text: str) -> None:
    pdf.ln(4)
    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"  {clean(text)}", fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*BLACK)
    pdf.ln(2)


def h3(pdf: ArticlePDF, text: str) -> None:
    pdf.ln(3)
    pdf.set_text_color(*BLUE)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*BLUE)
    pdf.set_line_width(0.25)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 70, pdf.get_y())
    pdf.set_text_color(*BLACK)
    pdf.ln(2)


def h4(pdf: ArticlePDF, text: str) -> None:
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 6, clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*BLACK)


def paragraph(pdf: ArticlePDF, text: str, indent: float = 0) -> None:
    text = strip_md(text)
    if not text:
        return
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*BLACK)
    pdf.set_x(pdf.l_margin + indent)
    pdf.multi_cell(
        pdf.w - pdf.l_margin - pdf.r_margin - indent,
        5.5, text,
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.ln(1)


def bullet(pdf: ArticlePDF, text: str, level: int = 0) -> None:
    indent = 6 + level * 5
    text = strip_md(text)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*BLACK)
    pdf.set_x(pdf.l_margin + indent)
    pdf.cell(5, 5.5, "-")
    pdf.set_x(pdf.l_margin + indent + 5)
    pdf.multi_cell(
        pdf.w - pdf.l_margin - pdf.r_margin - indent - 5,
        5.5, text,
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )


def code_block(pdf: ArticlePDF, code: str) -> None:
    pdf.ln(2)
    lines = code.strip().splitlines()
    block_h = len(lines) * 4.5 + 6
    if pdf.get_y() + block_h > pdf.h - pdf.b_margin:
        pdf.add_page()
    pdf.set_fill_color(*LGREY)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.2)
    pdf.rect(pdf.l_margin, pdf.get_y(),
             pdf.w - pdf.l_margin - pdf.r_margin, block_h, "FD")
    pdf.set_font("Courier", "", 7.5)
    pdf.set_text_color(50, 50, 50)
    for ln in lines:
        pdf.set_x(pdf.l_margin + 3)
        pdf.cell(0, 4.5, clean(ln[:100]),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*BLACK)
    pdf.ln(3)


def figure(pdf: ArticlePDF, fig_key: str, caption: str) -> None:
    png = FIGURES / FIG_MAP.get(fig_key, "")
    if not png.exists():
        return
    pdf.ln(4)
    if pdf.get_y() > 210:
        pdf.add_page()
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    img_w = min(usable_w, 155)
    img_x = pdf.l_margin + (usable_w - img_w) / 2
    pdf.image(str(png), x=img_x, y=pdf.get_y(), w=img_w)
    pdf.ln(img_w * 0.56 + 2)
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 5, strip_md(caption), align="C",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*BLACK)
    pdf.ln(4)


def table(pdf: ArticlePDF, table_lines: list[str]) -> None:
    rows: list[list[str]] = []
    for tl in table_lines:
        s = tl.strip()
        if re.match(r'^\|[-| :]+\|$', s):
            continue
        cells = [strip_md(c) for c in s.strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return

    max_cols = max(len(r) for r in rows)
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    if max_cols >= 4:
        col_w = [usable * 0.30] + [usable * 0.70 / (max_cols - 1)] * (max_cols - 1)
    elif max_cols == 3:
        col_w = [usable * 0.35, usable * 0.35, usable * 0.30]
    elif max_cols == 2:
        col_w = [usable * 0.42, usable * 0.58]
    else:
        col_w = [usable]

    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    pdf.ln(2)
    pdf._table_row = 0
    for idx, row in enumerate(rows):
        is_header = (idx == 0)
        fill = NAVY if is_header else (LGREY if pdf._table_row % 2 else WHITE)
        tc = WHITE if is_header else BLACK
        pdf.set_fill_color(*fill)
        pdf.set_text_color(*tc)
        pdf.set_font("Helvetica", "B" if is_header else "", 8)
        pdf.set_draw_color(*GREY)
        pdf.set_line_width(0.15)
        for cell, w in zip(row[:max_cols], col_w[:max_cols]):
            pdf.cell(w, 6, cell[:45], border=1, fill=True)
        pdf.ln(6)
        pdf._table_row += 1
    pdf.set_text_color(*BLACK)
    pdf.ln(3)


# ── Markdown renderer ─────────────────────────────────────────────────────────

def render_markdown(pdf: ArticlePDF, md_text: str) -> None:
    lines = md_text.splitlines()
    i = 0

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        # Skip page-break / front-matter dividers
        if line in ("---", "===", "***"):
            i += 1
            continue

        # H1 — already on title page
        if raw.startswith("# ") and not raw.startswith("## "):
            i += 1
            continue

        # H2
        if raw.startswith("## "):
            h2(pdf, raw[3:])
            i += 1
            continue

        # H3
        if raw.startswith("### "):
            h3(pdf, raw[4:])
            i += 1
            continue

        # H4
        if raw.startswith("#### "):
            h4(pdf, raw[5:])
            i += 1
            continue

        # Figure reference  ![alt](path)
        m = re.match(r'!\[([^\]]*)\]\(([^\)]+)\)', line)
        if m:
            alt = m.group(1)
            fig_key = next((k for k in FIG_MAP if k.lower() in alt.lower()), None)
            if fig_key:
                caption = alt
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("**Fig"):
                    caption = lines[i + 1].strip()
                    i += 1
                figure(pdf, fig_key, caption)
            i += 1
            continue

        # Table
        if line.startswith("|"):
            tbl: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            table(pdf, tbl)
            continue

        # Fenced code block
        if line.startswith("```"):
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # closing ```
            code_block(pdf, "\n".join(code))
            continue

        # Bullet / unordered list
        if re.match(r'^[-*•] ', line):
            bullet(pdf, line[2:], level=0)
            i += 1
            continue

        # Indented sub-bullet
        if re.match(r'^ {2,4}[-*] ', raw):
            bullet(pdf, re.sub(r'^ {2,4}[-*] ', '', raw), level=1)
            i += 1
            continue

        # Numbered list
        if re.match(r'^\d+\. ', line):
            bullet(pdf, re.sub(r'^\d+\. ', '', line))
            i += 1
            continue

        # Horizontal rule
        if re.match(r'^-{3,}$', line) or re.match(r'^\*{3,}$', line):
            pdf.ln(2)
            pdf.set_draw_color(*BLUE)
            pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
            i += 1
            continue

        # Empty line
        if not line:
            pdf.ln(2)
            i += 1
            continue

        # Blockquote
        if line.startswith(">"):
            paragraph(pdf, line.lstrip(">").strip(), indent=8)
            i += 1
            continue

        # Italic-only line  (*text*)
        if re.match(r'^\*[^*].+[^*]\*$', line):
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(*GREY)
            pdf.multi_cell(0, 5, clean(line.strip("*")),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(*BLACK)
            i += 1
            continue

        # Regular paragraph
        paragraph(pdf, raw)
        i += 1


# ── Entry point ───────────────────────────────────────────────────────────────

def build() -> None:
    print("Reading article...")
    md_text = ARTICLE.read_text(encoding="utf-8")

    print("Building PDF...")
    pdf = ArticlePDF()
    pdf.title_page()
    pdf.add_page()
    render_markdown(pdf, md_text)

    pdf.output(str(OUT))
    size_kb = OUT.stat().st_size / 1024
    print(f"PDF saved: {OUT}")
    print(f"Size: {size_kb:.0f} KB  |  Pages: {pdf.page}")


if __name__ == "__main__":
    build()
