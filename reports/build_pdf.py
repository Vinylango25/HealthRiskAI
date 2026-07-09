"""
reports/build_pdf.py
====================
Generates a publication-quality PDF of the HealthRisk AI research article.

    python reports/build_pdf.py

Output: reports/HealthRisk_AI_Research_Article.pdf
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent
ARTICLE = BASE / "healthrisk_ai_research_article.md"
FIGURES = BASE / "figures"
OUT     = BASE / "HealthRisk_AI_Research_Article.pdf"

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY   = (26,  73, 138)
BLUE   = (26, 115, 232)
GREY   = (90,  90,  90)
LGREY  = (245, 245, 245)
WHITE  = (255, 255, 255)
BLACK  = (30,  30,  30)
GREEN  = (52, 168,  83)
RED    = (234,  67,  53)

# ── Figure mapping (markdown alt-text → PNG file) ────────────────────────────
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


class ArticlePDF(FPDF):
    """Custom FPDF subclass for the research article layout."""

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(left=20, top=20, right=20)
        self._current_section = ""

    # ── Header / Footer ───────────────────────────────────────────────────────

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, "HealthRisk AI -- Bridging Clinical Intelligence and Financial Risk",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.set_draw_color(*BLUE)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, f"Page {self.page_no()}", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── Title page ────────────────────────────────────────────────────────────

    def title_page(self):
        self.add_page()

        # Top colour bar
        self.set_fill_color(*NAVY)
        self.rect(0, 0, self.w, 52, "F")

        # White title text
        self.set_y(12)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*WHITE)
        self.multi_cell(0, 10,
            "HealthRisk AI\nBridging Clinical Intelligence\nand Financial Risk",
            align="C")

        # Subtitle bar
        self.set_fill_color(*BLUE)
        self.rect(0, 52, self.w, 10, "F")
        self.set_y(54)
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(*WHITE)
        self.cell(0, 6, "A Multi-Model Empirical Study", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Author / date block
        self.set_y(75)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*NAVY)
        self.cell(0, 8, "Vincent Langat Kipkemoi", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*GREY)
        self.cell(0, 6, "HealthRisk Capital Partners  ·  July 2026",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Divider
        self.ln(6)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.5)
        self.line(40, self.get_y(), self.w - 40, self.get_y())
        self.ln(8)

        # Key results box
        self._kpi_box()

        # Bottom note
        self.set_y(-30)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 5,
            "Reproduce figures: python reports/generate_figures.py  |  "
            "Figures saved to reports/figures/",
            align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def _kpi_box(self):
        """Draw the key-results summary box on the title page."""
        x0, y0 = self.l_margin, self.get_y()
        bw = self.w - self.l_margin - self.r_margin

        # Box background
        self.set_fill_color(*LGREY)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.4)
        self.rect(x0, y0, bw, 62, "FD")

        # Box title
        self.set_xy(x0 + 4, y0 + 4)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*NAVY)
        self.cell(bw - 8, 7, "Key Results at a Glance",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        kpis = [
            ("Stacking Ensemble AUROC",       "0.831",  "target > 0.78  [PASS]", GREEN),
            ("Cost Prediction R2",            "0.28",   "+115% vs CMS-HCC baseline", BLUE),
            ("Cost Prediction MAPE",          "52%",    "target < 52%  [PASS]", GREEN),
            ("Hospital Default AUROC",        "0.851",  "Gini 0.702 - target > 0.50  [PASS]", GREEN),
            ("Survival C-index (Ensemble)",   "0.762",  "target > 0.70  [PASS]", GREEN),
            ("Actuarial Predictive Ratio",    "0.99",   "target 0.95-1.05  [PASS]", GREEN),
        ]

        col_w = bw / 3 - 2
        for i, (label, value, note, color) in enumerate(kpis):
            row = i // 3
            col = i % 3
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


# ── Save checkpoint ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Part 1 loaded OK — run build_pdf_full.py to generate the PDF")
