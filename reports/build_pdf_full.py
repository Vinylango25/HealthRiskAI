"""
reports/build_pdf_full.py
==========================
Full PDF builder — imports the ArticlePDF class and renders all content.

    python reports/build_pdf_full.py

Output: reports/HealthRisk_AI_Research_Article.pdf
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Re-use layout class defined in build_pdf.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_pdf import (
    ArticlePDF, ARTICLE, FIGURES, OUT, FIG_MAP,
    NAVY, BLUE, GREY, LGREY, WHITE, BLACK, GREEN, RED,
)

# ── Unicode sanitiser (fpdf2 with core fonts only supports latin-1) ───────────

_UNICODE_MAP = {
    "\u2713": "[PASS]", "\u2714": "[PASS]", "\u2718": "[FAIL]",
    "\u2019": "'",      "\u2018": "'",
    "\u201c": '"',      "\u201d": '"',
    "\u2014": "--",     "\u2013": "-",
    "\u00b2": "2",      "\u00b3": "3",
    "\u00b0": " deg",   "\u2265": ">=",     "\u2264": "<=",
    "\u00d7": "x",      "\u03b1": "alpha",  "\u03b2": "beta",
    "\u03bb": "lambda", "\u03c3": "sigma",  "\u2192": "->",
    "\u2190": "<-",     "\u00b1": "+/-",    "\u2248": "~=",
    "\u00e9": "e",      "\u00e8": "e",      "\u00ea": "e",
    "\u00e0": "a",      "\u00e2": "a",
}

def _clean(text: str) -> str:
    """Replace non-latin-1 characters with ASCII equivalents."""
    for char, repl in _UNICODE_MAP.items():
        text = text.replace(char, repl)
    # Final fallback: drop any remaining non-latin-1 chars
    return text.encode("latin-1", errors="replace").decode("latin-1")




def add_section_heading(pdf: ArticlePDF, text: str, level: int = 2):
    """Render a section heading (h2 or h3)."""
    text = _clean(text)
    pdf.ln(3)
    if level == 2:
        # Full-width coloured background
        pdf.set_fill_color(*NAVY)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"  {text}", fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        # Underlined sub-heading
        pdf.set_text_color(*BLUE)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(*BLUE)
        pdf.set_line_width(0.25)
        pdf.line(pdf.l_margin, pdf.get_y(),
                 pdf.l_margin + 60, pdf.get_y())
    pdf.set_text_color(*BLACK)
    pdf.ln(2)


def add_paragraph(pdf: ArticlePDF, text: str, indent: float = 0):
    """Render a body paragraph, handling **bold** inline markup."""
    if not text.strip():
        return
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*BLACK)

    # Strip inline markdown links but keep display text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = text.replace('`', '')
    text = _clean(text)

    # Split on **bold** tokens
    parts = re.split(r'(\*\*[^*]+\*\*)', text)
    x_start = pdf.l_margin + indent

    # Use multi_cell for simple paragraphs without bold
    if len(parts) == 1:
        pdf.set_x(x_start)
        pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - indent,
                       5.5, text.strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        # Write word by word, switching bold on/off
        full_line = ""
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                full_line += part[2:-2]   # keep content, mark handled later
            else:
                full_line += part
        # Simplest approach: render full paragraph with bold stripped
        pdf.set_x(x_start)
        pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - indent,
                       5.5, full_line.strip(),
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def add_bullet(pdf: ArticlePDF, text: str, level: int = 0):
    """Render a bullet point."""
    indent = 6 + level * 4
    bullet = "-"
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = text.replace('`', '').strip()
    text = _clean(text)

    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*BLACK)
    pdf.set_x(pdf.l_margin + indent)
    pdf.cell(5, 5.5, bullet)
    pdf.set_x(pdf.l_margin + indent + 5)
    pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - indent - 5,
                   5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def add_table_row(pdf: ArticlePDF, cells: list[str],
                  col_widths: list[float], header: bool = False):
    """Render one table row."""
    fill_color = NAVY if header else (WHITE if pdf._table_row % 2 == 0 else LGREY)
    text_color = WHITE if header else BLACK
    pdf.set_fill_color(*fill_color)
    pdf.set_text_color(*text_color)
    pdf.set_font("Helvetica", "B" if header else "", 8)
    pdf.set_draw_color(*GREY)
    pdf.set_line_width(0.15)

    row_h = 6
    for i, (cell, w) in enumerate(zip(cells, col_widths)):
        cell = re.sub(r'\*\*([^*]+)\*\*', r'\1', str(cell)).strip()
        cell = cell.replace('`', '')
        cell = _clean(cell)
        pdf.cell(w, row_h, cell[:40], border=1, fill=True)
    pdf.ln(row_h)
    if not hasattr(pdf, '_table_row'):
        pdf._table_row = 0
    pdf._table_row += 1


def add_figure(pdf: ArticlePDF, fig_key: str, caption: str):
    """Insert a figure image with caption."""
    png = FIGURES / FIG_MAP.get(fig_key, "")
    if not png.exists():
        return
    pdf.ln(3)
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    img_w = min(usable_w, 160)
    img_x = pdf.l_margin + (usable_w - img_w) / 2

    # Check space — add page if needed
    if pdf.get_y() > 220:
        pdf.add_page()

    pdf.image(str(png), x=img_x, y=pdf.get_y(), w=img_w)
    pdf.ln(img_w * 0.55 + 2)  # approximate height from aspect ratio

    # Caption
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(*GREY)
    clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', caption)
    pdf.multi_cell(0, 5, clean, align="C",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*BLACK)
    pdf.ln(4)


def add_code_block(pdf: ArticlePDF, code: str):
    """Render a monospaced code block."""
    pdf.ln(2)
    pdf.set_fill_color(*LGREY)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.2)
    lines = code.strip().splitlines()
    block_h = len(lines) * 4.5 + 4
    pdf.rect(pdf.l_margin, pdf.get_y(), pdf.w - pdf.l_margin - pdf.r_margin,
             block_h, "FD")
    pdf.set_font("Courier", "", 7.5)
    pdf.set_text_color(50, 50, 50)
    for line in lines:
        pdf.set_x(pdf.l_margin + 3)
        pdf.cell(0, 4.5, line[:90], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*BLACK)
    pdf.ln(3)


# ── Markdown parser ───────────────────────────────────────────────────────────

def render_markdown(pdf: ArticlePDF, md_text: str):
    """Parse markdown line-by-line and render with the PDF helpers."""
    lines = md_text.splitlines()
    i = 0
    pdf._table_row = 0

    while i < len(lines):
        line = lines[i]

        # Skip YAML / page-break markers
        if line.strip() in ("---", "==="):
            i += 1
            continue

        # H1 (title — already on title page)
        if line.startswith("# ") and not line.startswith("## "):
            i += 1
            continue

        # H2
        if line.startswith("## "):
            text = line[3:].strip()
            # Skip abstract heading (rendered inline)
            add_section_heading(pdf, text, level=2)
            i += 1
            continue

        # H3
        if line.startswith("### "):
            add_section_heading(pdf, line[4:].strip(), level=3)
            i += 1
            continue

        # H4
        if line.startswith("#### "):
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.set_text_color(*NAVY)
            pdf.cell(0, 6, line[5:].strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(*BLACK)
            i += 1
            continue

        # Figure image reference  ![caption](path)
        img_match = re.match(r'!\[([^\]]*)\]\(([^\)]+)\)', line.strip())
        if img_match:
            alt = img_match.group(1)
            fig_key = None
            for k in FIG_MAP:
                if k.lower() in alt.lower():
                    fig_key = k
                    break
            if fig_key:
                # Look ahead for caption italic line
                caption = alt
                if i + 1 < len(lines) and lines[i+1].startswith("**Fig"):
                    caption = lines[i+1]
                    i += 1
                add_figure(pdf, fig_key, caption)
            i += 1
            continue

        # Table (starts with |)
        if line.strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            _render_table(pdf, table_lines)
            continue

        # Code block
        if line.strip().startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            add_code_block(pdf, "\n".join(code_lines))
            continue

        # Bullet
        if line.strip().startswith(("- ", "* ", "• ")):
            add_bullet(pdf, line.strip()[2:])
            i += 1
            continue

        # Numbered list
        if re.match(r'^\d+\. ', line.strip()):
            add_bullet(pdf, re.sub(r'^\d+\. ', '', line.strip()))
            i += 1
            continue

        # Horizontal rule
        if re.match(r'^-{3,}$', line.strip()):
            pdf.ln(3)
            pdf.set_draw_color(*BLUE)
            pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin, pdf.get_y(),
                     pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
            i += 1
            continue

        # Empty line
        if not line.strip():
            pdf.ln(2)
            i += 1
            continue

        # Italic metadata line (*Author · Date*)
        if line.strip().startswith("*") and line.strip().endswith("*"):
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(*GREY)
            pdf.multi_cell(0, 5, line.strip().strip("*"),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(*BLACK)
            i += 1
            continue

        # Regular paragraph
        add_paragraph(pdf, line)
        i += 1


def _render_table(pdf: ArticlePDF, table_lines: list[str]):
    """Parse and render a markdown table."""
    rows = []
    for tl in table_lines:
        if re.match(r'^\|[-| :]+\|$', tl.strip()):
            continue  # skip separator line
        cells = [c.strip() for c in tl.strip().strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return

    max_cols = max(len(r) for r in rows)
    usable = pdf.w - pdf.l_margin - pdf.r_margin

    # Dynamic column widths: first col wider
    if max_cols >= 4:
        col_w = [usable * 0.32] + [usable * 0.68 / (max_cols - 1)] * (max_cols - 1)
    elif max_cols == 3:
        col_w = [usable * 0.35, usable * 0.35, usable * 0.30]
    elif max_cols == 2:
        col_w = [usable * 0.45, usable * 0.55]
    else:
        col_w = [usable]

    # Pad rows
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    pdf.ln(2)
    pdf._table_row = 0
    for idx, row in enumerate(rows):
        add_table_row(pdf, row[:max_cols], col_w[:max_cols], header=(idx == 0))
    pdf.set_text_color(*BLACK)
    pdf.ln(3)


# ── Main ──────────────────────────────────────────────────────────────────────

def build():
    md_text = ARTICLE.read_text(encoding="utf-8")

    pdf = ArticlePDF()
    pdf.title_page()
    pdf.add_page()

    render_markdown(pdf, md_text)

    pdf.output(str(OUT))
    print(f"✅  PDF saved: {OUT}")
    print(f"    Size: {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build()
