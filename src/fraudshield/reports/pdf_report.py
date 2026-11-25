"""PDF report built from a ``Summary``.

Charts are drawn with Matplotlib (static images suit print), the document with
ReportLab. Every number in the PDF comes from the Summary object; nothing is
written in by hand.
"""

from __future__ import annotations

import io
from html import escape

import matplotlib

matplotlib.use("Agg")  # draw to images, no window
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from fraudshield import __version__  # noqa: E402
from fraudshield.config import SETTINGS  # noqa: E402
from fraudshield.reports.summary import Summary, risk_level_text  # noqa: E402
from fraudshield.ui import theme  # noqa: E402

# Print palette: same meaning as the app, darker shades for white paper.
TEAL = colors.HexColor("#0F766E")
INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#6B7280")
LINE = colors.HexColor("#E5E7EB")
SOFT = colors.HexColor("#F3F4F6")
RISK_PRINT = {"Low": "#16A34A", "Medium": "#D97706", "High": "#EA580C", "Critical": "#DC2626"}

DISCLAIMER = (
    "Fraud probabilities are model estimates based on past patterns. They help decide "
    "which transactions to review first; they are not confirmed fraud decisions."
)


def _text(value) -> str:
    """Escape for ReportLab's mini-HTML and keep to characters the PDF font has."""
    s = escape(str(value))
    return s.replace("≥", "&gt;=").replace("≤", "&lt;=").replace("→", "-&gt;")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "t", parent=base["Title"], fontSize=20, textColor=INK, alignment=0, spaceAfter=4
        ),
        "subtitle": ParagraphStyle("s", parent=base["Normal"], fontSize=9.5, textColor=MUTED),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=13, textColor=TEAL, spaceBefore=14, spaceAfter=6
        ),
        "body": ParagraphStyle(
            "b", parent=base["Normal"], fontSize=9.5, leading=13.5, textColor=INK
        ),
        "small": ParagraphStyle(
            "sm", parent=base["Normal"], fontSize=8, leading=11, textColor=MUTED
        ),
        "bullet": ParagraphStyle(
            "bl",
            parent=base["Normal"],
            fontSize=9.5,
            leading=13.5,
            textColor=INK,
            leftIndent=12,
            bulletIndent=2,
        ),
        "kpi_label": ParagraphStyle("kl", parent=base["Normal"], fontSize=8, textColor=MUTED),
        "kpi_value": ParagraphStyle(
            "kv",
            parent=base["Normal"],
            fontSize=14,
            leading=17,
            textColor=INK,
            fontName="Helvetica-Bold",
        ),
    }


def _table(rows: list[list], widths: list[float], header: bool = True) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), SOFT),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    table.setStyle(TableStyle(style))
    return table


def _image(fig, width_cm: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width_cm * cm, height=width_cm * cm * h / w)


def _axes(fig_w: float, fig_h: float):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(colors="#374151", labelsize=8)
    ax.grid(axis="y", color="#F3F4F6")
    ax.set_axisbelow(True)
    return fig, ax


def tier_chart(summary: Summary):
    fig, ax = _axes(6.5, 2.3)
    tiers = [t["tier"] for t in summary.tiers]
    counts = [max(t["transactions"], 0) for t in summary.tiers]
    bars = ax.bar(tiers, counts, color=[RISK_PRINT[t] for t in tiers], width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)", fontsize=8, color="#374151")
    ax.set_title("Transactions per risk tier", fontsize=10, loc="left", color="#111827")
    for bar, n in zip(bars, counts, strict=True):
        if n:
            ax.annotate(
                f"{n:,}",
                (bar.get_x() + bar.get_width() / 2, n),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#111827",
            )
    return fig


def weekly_chart(summary: Summary):
    weekly = summary.weekly
    fig, ax = _axes(6.5, 2.3)
    ax.plot(
        weekly["week"], weekly["transactions"], color="#9CA3AF", lw=1.5, label="All transactions"
    )
    ax.set_ylabel("Transactions per week", fontsize=8, color="#374151")
    ax.set_ylim(bottom=0)  # start at zero so small weekly changes are not exaggerated
    ax2 = ax.twinx()
    column, label = ("fraud", "Known fraud") if "fraud" in weekly else ("flagged", "Flagged")
    ax2.bar(weekly["week"], weekly[column], width=5, color=theme.FRAUD, alpha=0.75, label=label)
    ax2.set_ylabel(f"{label} per week", fontsize=8, color=theme.FRAUD)
    ax2.tick_params(colors=theme.FRAUD, labelsize=8)
    ax2.yaxis.get_major_locator().set_params(integer=True)
    for side in ("top",):
        ax2.spines[side].set_visible(False)
    ax.set_title("Weekly activity", fontsize=10, loc="left", color="#111827")
    fig.autofmt_xdate()
    return fig


def build_pdf(summary: Summary) -> bytes:
    s = _styles()
    story = []
    ds, sc = summary.dataset, summary.scoring

    story += [
        Paragraph("FraudShield AI - Fraud Analysis Report", s["title"]),
        Paragraph(
            f"Dataset: <b>{_text(ds['name'])}</b> &nbsp;·&nbsp; {ds['start']} to {ds['end']} "
            f"&nbsp;·&nbsp; Model: {_text(summary.model['name'])} (trained "
            f"{summary.model['trained']}) &nbsp;·&nbsp; Generated {summary.generated_at} "
            f"&nbsp;·&nbsp; FraudShield AI v{__version__}",
            s["subtitle"],
        ),
        Spacer(1, 8),
        Paragraph(_text(DISCLAIMER), s["small"]),
    ]

    # --- Key figures -------------------------------------------------------------
    kpis = [
        ("Transactions analysed", f"{ds['transactions']:,}"),
        ("Total value", f"{summary.totals['total_amount']:,.0f}"),
        ("Flagged for review", f"{sc['flagged']:,} ({sc['flagged_share']:.2%})"),
        ("Value flagged", f"{sc['flagged_amount']:,.0f}"),
        ("High + Critical risk", f"{sc['high_or_critical']:,}"),
        (
            "Data quality score",
            f"{ds['quality_score']:.1f}%" if ds["quality_score"] is not None else "-",
        ),
    ]
    if summary.labelled:
        lab = summary.labelled
        kpis += [
            ("Known fraud", f"{lab['known_fraud']:,} ({lab['known_fraud_rate']:.2%})"),
            ("Known fraud flagged", f"{lab['recall']:.1%}" if lab["recall"] is not None else "-"),
        ]
    cells = [
        [Paragraph(_text(label), s["kpi_label"]), Paragraph(_text(value), s["kpi_value"])]
        for label, value in kpis
    ]
    grid = [cells[i : i + 4] for i in range(0, len(cells), 4)]
    grid = [[Table([[c[0]], [c[1]]], colWidths=[4.1 * cm]) for c in row] for row in grid]
    kpi_table = Table(grid, colWidths=[4.3 * cm] * 4)
    kpi_table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story += [
        Paragraph("Key figures", s["h2"]),
        kpi_table,
        Spacer(1, 6),
        Paragraph(_text(risk_level_text(summary)), s["body"]),
    ]

    # --- Findings ------------------------------------------------------------------
    story.append(Paragraph("Key findings", s["h2"]))
    for finding in summary.findings:
        story.append(Paragraph(_text(finding), s["bullet"], bulletText="•"))
    for warning in summary.warnings:
        story.append(Paragraph(f"<b>Note:</b> {_text(warning)}", s["small"]))

    # --- Risk distribution ------------------------------------------------------------
    rt = SETTINGS.risk_thresholds
    ranges = {
        "Low": f"below {rt.medium:.0%}",
        "Medium": f"{rt.medium:.0%}-{rt.high:.0%}",
        "High": f"{rt.high:.0%}-{rt.critical:.0%}",
        "Critical": f"{rt.critical:.0%} and above",
    }
    rows = [["Tier", "Probability", "Action", "Transactions", "Share", "Amount", "Known fraud"]]
    for t in summary.tiers:
        rows.append(
            [
                t["tier"],
                ranges[t["tier"]],
                t["action"],
                f"{t['transactions']:,}",
                f"{t['share']:.2%}",
                f"{t['amount']:,.0f}",
                f"{t['known_fraud_rate']:.1%}" if t["known_fraud_rate"] is not None else "-",
            ]
        )
    story += [
        Paragraph("Risk distribution", s["h2"]),
        _table(rows, [2 * cm, 2.6 * cm, 3.9 * cm, 2.3 * cm, 1.6 * cm, 2.3 * cm, 2.2 * cm]),
        Spacer(1, 6),
        _image(tier_chart(summary), 15),
        Paragraph(
            "Tiers come from calibrated probabilities with configurable cut-offs. A tier's "
            "known fraud rate should roughly match its probability range.",
            s["small"],
        ),
    ]

    story += [
        KeepTogether([Paragraph("Activity over time", s["h2"]), _image(weekly_chart(summary), 15)])
    ]

    # --- Categories -----------------------------------------------------------------------
    if summary.top_categories:
        rows = [
            ["Category", "High-risk transactions", "Share of high risk", "Share of all", "Lift"]
        ]
        rows += [
            [
                _text(c["category"]),
                f"{c['high_risk_transactions']:,}",
                f"{c['share_of_high_risk']:.1%}",
                f"{c['share_of_all']:.1%}",
                f"{c['lift']:.1f}x",
            ]
            for c in summary.top_categories
        ]
        story += [
            KeepTogether(
                [
                    Paragraph("Categories over-represented among high-risk transactions", s["h2"]),
                    _table(rows, [4.5 * cm, 3.6 * cm, 3.3 * cm, 2.8 * cm, 2 * cm]),
                    Paragraph(
                        "Lift = share among High/Critical transactions divided by share among all "
                        "transactions. Categories under 0.5% of transactions are left out.",
                        s["small"],
                    ),
                ]
            )
        ]

    # --- Model performance --------------------------------------------------------------------
    m = summary.model
    rows = [
        ["Metric", "Value", "Meaning"],
        [
            "PR-AUC",
            f"{m['pr_auc']:.3f}",
            f"Ranking quality for rare fraud (random guessing: {m['fraud_rate']:.3f})",
        ],
        ["Recall", f"{m['recall']:.1%}", "Share of fraud caught"],
        ["Precision", f"{m['precision']:.1%}", "Share of alerts that were fraud"],
        ["F1", f"{m['f1']:.3f}", "Balance of precision and recall"],
        ["ROC-AUC", f"{m['roc_auc']:.3f}", "Ranking quality (optimistic when fraud is rare)"],
        ["Decision threshold", f"{m['threshold']:.2f}", "Chosen on validation data"],
    ]
    comparison = [["Model", "Test PR-AUC", "Fraud caught at same alert volume"]]
    comparison += [
        [
            c["model"] + (" (selected)" if c["model"] == m["name"] else ""),
            f"{c['pr_auc']:.3f}",
            f"{c['recall_at_same_alerts']:.1%}",
        ]
        for c in m["comparison"]
    ]
    story += [
        *(
            [
                Paragraph("Model performance", s["h2"]),
                Paragraph(
                    f"Selected model: <b>{_text(m['name'])}</b>. Measured once on "
                    f"{m['test_transactions']:,} transactions from {m['test_period']} that the "
                    "model never saw while learning (time-based split).",
                    s["body"],
                ),
                Spacer(1, 4),
                _table(rows, [3.6 * cm, 2.4 * cm, 10.6 * cm]),
                Spacer(1, 8),
                _table(comparison, [6.2 * cm, 3 * cm, 7.4 * cm]),
                Paragraph(
                    "Accuracy is not used to judge models: with fraud this rare, approving every "
                    "transaction would already look over 99% accurate.",
                    s["small"],
                ),
            ]
        )
    ]

    buf = io.BytesIO()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, 1.2 * cm, f"FraudShield AI · {summary.generated_at}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2 * cm,
        title="FraudShield AI - Fraud Analysis Report",
        author="FraudShield AI",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
