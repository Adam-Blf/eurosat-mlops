"""Assemble the PDF report: cover, contents, then the sections, from the committed results."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fpdf.enums import XPos, YPos

from eurosat_mlops import __version__
from eurosat_mlops.report import charts, sections_advanced, sections_levels, sections_model
from eurosat_mlops.report.engine import (
    ASSETS,
    DARK,
    GREY,
    LIGHT,
    MARGIN,
    NAVY,
    ROOT,
    ROSE,
    WHITE,
    Report,
)
from eurosat_mlops.report.results import Results, fr, load, pct, thousands

TITLE = "Mise en production d'un classifieur d'images satellite"
SHORT_TITLE = "eurosat-mlops, Beloucif et Morice, EFREI, cours MLOps"
OUTPUT = ROOT / "reports" / "rapport-eurosat-mlops.pdf"
FIGURES = ROOT / "reports" / "figures"
MONTHS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]  # fmt: skip

SECTIONS = [
    sections_model.context,
    sections_model.architecture,
    sections_levels.level0,
    sections_levels.level1,
    sections_levels.level2,
    sections_advanced.level3,
    sections_advanced.quality,
    sections_advanced.security,
    sections_advanced.closing,
]

COVERAGE = [
    ["0", "Stack complète en une commande, en conteneurs", "Réalisé, rejoué par la CI"],
    ["1", "Déploiement automatique (commande ou CD)", "Écrit et testé, aucune VM réelle"],
    ["1", "Accès refusé sans jeton dans l'en-tête", "Réalisé"],
    ["1", "Usage suivi par utilisateur", "Réalisé"],
    ["1", "Test de charge", "Réalisé, sur la stack locale"],
    ["2", "VM créée par Terraform", "Validé, jamais appliqué"],
    ["2", "Interface utilisateur", "Réalisé"],
    ["2", "Détection de dérive", "Réalisé, démo sur images dégradées"],
    ["2", "MLflow pour versionner et promouvoir", "Réalisé"],
    ["3", "Quantization", "Réalisé, int8 statique servi"],
    ["3", "Scaling horizontal", "Réalisé, mesuré en local"],
    ["3", "Canary sur Kubernetes", "Manifestes écrits, non éprouvés"],
]


def _figures(results: Results) -> dict[str, Path]:
    return {
        "architecture": charts.architecture(FIGURES / "architecture.png"),
        "confusion": charts.confusion(results, FIGURES / "confusion.png"),
        "quantization": charts.quantization(results, FIGURES / "quantization.png"),
        "scaling": charts.scaling(results, FIGURES / "scaling.png"),
        "drift": charts.drift(results, FIGURES / "drift.png"),
    }


def _cover(pdf: Report, results: Results) -> None:
    pdf.add_page()
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, pdf.w, 74, "F")
    pdf.set_fill_color(*ROSE)
    pdf.rect(0, 74, pdf.w, 2.2, "F")
    pdf.image(str(ASSETS / "efrei-logo-blanc.png"), x=MARGIN, y=13, h=12)
    pdf.set_xy(MARGIN, 36)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Display", "", 22)
    pdf.multi_cell(pdf.w - 2 * MARGIN, 10, TITLE, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
    pdf.ln(1.5)
    pdf.set_font("Body", "", 10)
    pdf.multi_cell(
        0, 5, "Projet MLOps sur EuroSAT : API à jeton, MLflow, dérive, Terraform et Kubernetes",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L",
    )  # fmt: skip
    pdf.set_y(83)
    pdf.set_text_color(*DARK)
    pdf.set_font("Body", "B", 10.5)
    pdf.cell(0, 5.5, "Adam Beloucif et Emilien Morice", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Body", "", 9)
    pdf.set_text_color(*GREY)
    today = date.today()
    pdf.multi_cell(
        0, 4.6,
        "Cours MLOps, Mastère Data Engineering et IA, EFREI Paris, 2026-2027.\n"
        f"Version {__version__} du dépôt Adam-Blf/eurosat-mlops, "
        f"{today.day} {MONTHS[today.month - 1]} {today.year}.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )  # fmt: skip
    pdf.ln(4)
    pdf.set_font("Body", "B", 10.5)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 6, "Résumé", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.para(
        "Nous avons mis en production un classifieur d'images satellite (EuroSAT, 10 classes) "
        "pour le cours MLOps. Le modèle, un ResNet18 affiné puis quantifié en int8, est servi par "
        "une API FastAPI protégée par jeton, versionné dans MLflow, surveillé par un détecteur de "
        "dérive et lancé en conteneurs, sur une VM décrite par Terraform ou sur Kubernetes avec un "
        "canary.",
        size=9.2,
    )
    pdf.para(
        "Ce rapport suit les niveaux du sujet, donne les mesures obtenues et dit ce qui n'a pas "
        "été fait. Aucune VM de production réelle n'a été déployée, le test de charge a tourné "
        "contre la stack locale et les manifestes Kubernetes n'ont pas été éprouvés sur un "
        "cluster.",
        size=9.2,
    )
    _key_figures(pdf, results)
    pdf.set_font("Body", "B", 10.5)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 6, "Couverture du sujet", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.data_table(
        [["Niveau", "Exigence", "État"], *COVERAGE], [16, 90, 70], "Niveaux du sujet et état réel.",
        size=7.8, align="CLL",
    )  # fmt: skip


def _key_figures(pdf: Report, results: Results) -> None:
    served = results.quant["int8 static"]
    total = sum(int(r["requests"]) for r in results.scaling)
    failures = sum(int(r["failures"]) for r in results.scaling)
    boxes = [
        (pct(served["accuracy"]), "accuracy du modèle servi"),
        (fr(served["size_mb"], 1) + " Mo", "taille du modèle servi"),
        (fr(results.quant["fp32"]["median_ms"] / served["median_ms"], 2) + "x",
         "plus rapide que le fp32"),
        (str(failures), f"échec sur {thousands(total)} requêtes"),
    ]  # fmt: skip
    gap, width, height = 4, (pdf.w - 2 * MARGIN - 12) / 4, 21
    y = pdf.get_y() + 1
    for i, (value, label) in enumerate(boxes):
        x = MARGIN + i * (width + gap)
        pdf.set_fill_color(*LIGHT)
        pdf.rect(x, y, width, height, "F")
        pdf.set_fill_color(*ROSE)
        pdf.rect(x, y, 1.2, height, "F")
        pdf.set_xy(x + 3, y + 3)
        pdf.set_font("Display", "", 15)
        pdf.set_text_color(*NAVY)
        pdf.cell(width - 4, 7, value)
        pdf.set_xy(x + 3, y + 12)
        pdf.set_font("Body", "", 7.6)
        pdf.set_text_color(*GREY)
        pdf.multi_cell(width - 4, 3.6, label)
    pdf.set_y(y + height + 5)


def _contents(pdf: Report, entries: list[tuple[str, int]]) -> None:
    pdf.set_font("Display", "", 15)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 9, "Sommaire", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    for title, page in entries:
        pdf.set_font("Body", "", 9.6)
        pdf.set_text_color(*DARK)
        width = pdf.w - 2 * MARGIN - 12
        pdf.cell(width, 5.4, title, link=pdf.add_link(page=page or 1))
        pdf.cell(12, 5.4, str(page), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)


def _assemble(results: Results, figs: dict[str, Path], entries: list[tuple[str, int]]) -> Report:
    pdf = Report(TITLE, SHORT_TITLE)
    _cover(pdf, results)
    pdf.add_page()
    _contents(pdf, entries)
    for section in SECTIONS:
        section(pdf, results, figs)
    pdf.finish()
    return pdf


def build_pdf(out: Path = OUTPUT) -> Path:
    """Write the report. Passes repeat until the contents page matches the real pagination."""
    results = load()
    figs = _figures(results)
    entries: list[tuple[str, int]] = [("", 0)]
    for _ in range(4):
        pdf = _assemble(results, figs, entries)
        found = [(s.name, s.page_number) for s in pdf._outline if s.level == 0]  # noqa: SLF001
        if found == entries:
            out.parent.mkdir(parents=True, exist_ok=True)
            pdf.output(str(out))
            return out
        entries = found
    raise RuntimeError("the contents page never settled")
