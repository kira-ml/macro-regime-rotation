"""
Generate an academic-style PDF report for the Macro-Informed Sector Rotation project.
Uses reportlab for PDF generation with embedded figures and formatted tables.
"""

import os
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    Image, ListFlowable, ListItem
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_PDF = "Macro_Regime_Rotation_Report.pdf"
IMAGE_DIR = r"D:\machine_learning_prototypes\macro-regime-rotation\outputs"

IMAGES = {
    "hero": os.path.join(IMAGE_DIR, "hero_chart.png"),
    "heatmap": os.path.join(IMAGE_DIR, "regime_heatmap.png"),
    "drawdown": os.path.join(IMAGE_DIR, "drawdowns.png"),
    "rolling_sharpe": os.path.join(IMAGE_DIR, "rolling_sharpe.png"),
    "timeline": os.path.join(IMAGE_DIR, "regime_timeline_events.png"),
}

# ============================================================================
# PDF GENERATION LOGIC
# ============================================================================

def generate_pdf():
    doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=LETTER,
                            leftMargin=1*inch, rightMargin=1*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    
    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        'Title', parent=styles['Title'], fontName='Times-Bold', fontSize=18, 
        alignment=TA_CENTER, spaceAfter=12
    )
    author_style = ParagraphStyle(
        'Author', parent=styles['Normal'], fontName='Times-Roman', fontSize=11, 
        alignment=TA_CENTER, spaceAfter=20
    )
    heading_style = ParagraphStyle(
        'Heading', parent=styles['Heading2'], fontName='Times-Bold', fontSize=13, 
        spaceAfter=6, spaceBefore=12
    )
    subheading_style = ParagraphStyle(
        'Subheading', parent=styles['Heading3'], fontName='Times-Bold', fontSize=11, 
        spaceAfter=4, spaceBefore=6
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'], fontName='Times-Roman', fontSize=10.5, 
        alignment=TA_JUSTIFY, spaceAfter=6
    )
    caption_style = ParagraphStyle(
        'Caption', parent=styles['Normal'], fontName='Times-Italic', fontSize=9.5, 
        alignment=TA_CENTER, spaceAfter=10
    )

    # Build Document Elements
    story = []

    # --- Title ---
    story.append(Paragraph("Macro-Informed Sector Rotation", title_style))
    story.append(Paragraph("A Framework for Regime-Aware Tactical Asset Allocation", author_style))
    story.append(Paragraph("Ken Ira Lacson Talingting", author_style))
    story.append(Spacer(1, 0.15*inch))

    # --- Section 1: Problem Framing ---
    story.append(Paragraph("I. Problem Selection &amp; Framing", heading_style))
    story.append(Paragraph(
        "Static allocation rules are known to perform poorly during regime shifts—periods where the underlying data-generating process changes structurally. "
        "Instead of treating tactical asset allocation as a pure return-forecasting problem, this project reframes it as a <b>regime detection problem</b>. "
        "The objective is to infer the latent macroeconomic environment from observable data and condition sector selection on that inferred state. "
        "Regime detection is a more tractable unsupervised learning task than noisy return prediction, and the resulting allocation logic is transparent and interpretable.",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    # --- Section 2: Constraints & Assumptions ---
    story.append(Paragraph("II. Constraints, Assumptions &amp; Design Choices", heading_style))
    
    story.append(Paragraph("<b>Key Assumptions</b>", subheading_style))
    story.append(Paragraph(
        "1. <b>Stationarity within regimes:</b> The approach assumes that while the market shifts between discrete states, the return distributions of sectors are relatively stable within each state. "
        "This is a strong assumption but necessary for a discrete-state HMM.<br/>"
        "2. <b>Feature sufficiency:</b> The approach assumes that a compact set of macro variables (yield curve, credit spreads, volatility) contains enough signal to identify regime shifts. "
        "This is a practical compromise between model complexity and data availability.<br/>"
        "3. <b>Discrete states:</b> The continuous spectrum of market conditions is modeled as 3 discrete states. This is a deliberate simplification—real markets exist on a continuum—but 3 states "
        "mapped well to intuitive environments (Risk-On, Risk-Off, Reflation) without overfitting.",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>Data Considerations</b>", subheading_style))
    story.append(Paragraph(
        "Data was sourced exclusively from yfinance to ensure reproducibility. The universe comprises 11 GICS sector ETFs and 10 macro proxies. "
        "A dynamic availability mask handles the later inception dates of XLRE (2015) and XLC (2018), ensuring the strategy only allocates to sectors that existed at the time.",
        body_style
    ))
    story.append(Paragraph(
        "Feature engineering was designed to prevent look-ahead bias. Features at time <i>t</i> are constructed using only data available at or before <i>t</i>. "
        "Raw features are shifted by 1 month before any standardization, and standardization is performed per-fold during walk-forward validation.",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    # --- Section 3: Methodology ---
    story.append(Paragraph("III. Methodology: Models &amp; Validation", heading_style))
    
    story.append(Paragraph("<b>Model Selection</b>", subheading_style))
    story.append(Paragraph(
        "A Hidden Markov Model (HMM) was chosen over simpler static clustering (GMM) because regimes are inherently persistent. "
        "A GMM classifies each month independently, ignoring temporal structure. The HMM explicitly models the transition matrix, capturing the fact that if the market is in a Risk-Off regime today, "
        "it is likely to remain there next month.",
        body_style
    ))
    story.append(Paragraph(
        "The HMM was trained with Gaussian emissions. A diagonal transition prior (<font face='Courier'>transmat_prior=10.0</font>) was applied to enforce state persistence. "
        "Without this prior, the model produced regimes that flickered between states month-to-month, generating approximately 62% monthly turnover—economically implausible. "
        "The prior reduced turnover to 16% while preserving the model's ability to detect genuine regime shifts at major inflection points (COVID, 2022 rate hikes, SVB collapse).",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>Regime Count Selection</b>", subheading_style))
    story.append(Paragraph(
        "The choice of 3 regimes was validated by testing 2, 3, and 4 regimes with both GMM and HMM. "
        "Two regimes collapsed to a simple risk-on/risk-off binary that missed the reflationary dynamics of 2022. "
        "Four regimes produced a fragmented state with minimal occurrences, offering little practical value. "
        "Three regimes provided the best balance of economic interpretability and statistical stability.",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>Evaluation Strategy</b>", subheading_style))
    story.append(Paragraph(
        "The strategy was evaluated using walk-forward validation: 5 years of initial training (Aug 2018 – Jul 2023), followed by annual refits on an expanding window. "
        "The out-of-sample period spans 29 months (Aug 2023 – Dec 2025). "
        "Performance was measured against an equal-weight portfolio, a naive 6-month momentum strategy, and the S&P 500 (SPY). "
        "All backtests include a 5 basis point one-way transaction cost.",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    # --- Section 4: Cumulative Returns Chart ---
    if os.path.exists(IMAGES["hero"]):
        img = Image(IMAGES["hero"], width=7*inch, height=4.5*inch)
        story.append(img)
        story.append(Paragraph(
            "Figure 1: Out-of-sample cumulative returns (Aug 2023 – Dec 2025, 29 months). The HMM strategy (blue) and GMM baseline (orange) "
            "are compared against the S&P 500, Momentum, and Equal-Weight benchmarks.",
            caption_style
        ))
        story.append(Spacer(1, 0.15*inch))

    # --- Section 5: Results ---
    story.append(Paragraph("IV. Results", heading_style))
    story.append(Paragraph(
        "The table below presents the out-of-sample performance metrics. The HMM strategy delivered a Sharpe ratio of 1.64 over the evaluation period, "
        "with an annualized return of 25.53% and a maximum drawdown of -6.68%. "
        "The out-of-sample period (Aug 2023 – Dec 2025) was characterized by a strong AI-driven bull market, which may not generalize to other environments.",
        body_style
    ))
    
    # The Metrics Table
    data = [
        ["Metric", "HMM (3 States)", "GMM", "SPY", "Momentum"],
        ["Ann. Return", "25.53%", "26.29%", "14.58%", "14.09%"],
        ["Ann. Volatility", "12.69%", "14.18%", "16.82%", "16.35%"],
        ["Sharpe Ratio", "1.64", "1.52", "0.72", "0.71"],
        ["Max Drawdown", "-6.68%", "-10.23%", "-23.93%", "-16.61%"],
        ["Turnover", "16.09%", "2.30%", "0.00%", "56.32%"]
    ]
    
    table = Table(data, colWidths=[1.8*inch, 1.1*inch, 1.1*inch, 1.1*inch, 1.1*inch])
    table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('FONTNAME', (0,0), (0,-1), 'Times-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.05*inch))
    story.append(Paragraph(
        "<i>Note: Sharpe ratios use the 3-month T-bill rate (^IRX) from the same data source, averaged over the full sample period (1.62% annualized), as the risk-free rate.</i>",
        caption_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "The static GMM slightly outperformed the HMM in raw annualized return (26.29% vs. 25.53%). "
        "However, the HMM delivered a higher Sharpe ratio (1.64 vs. 1.52) and a smaller maximum drawdown (-6.68% vs. -10.23%). "
        "This pattern—where static models capture persistent trends effectively but with higher volatility and deeper drawdowns— "
        "suggests that the HMM's temporal modeling was associated with lower volatility and smaller drawdowns in this sample, "
        "even though it did not improve raw returns in trending markets.",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>Regime-Conditional Performance</b>", subheading_style))
    story.append(Paragraph(
        "The HMM's regime-conditional Sharpe ratios were 1.77 for State 0 (Risk-On) and 1.59 for State 1 (Risk-Off). "
        "State 2 (Reflation) was identified as a rare state, occurring only once in the 29-month out-of-sample window, "
        "so its conditional Sharpe could not be reliably estimated. "
        "The model identified distinct environments with different risk-reward profiles, "
        "though the short evaluation period limits the statistical confidence in these conditional estimates.",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    # --- Section 6: Visual Analysis ---
    story.append(Paragraph("V. Visual Analysis", heading_style))

    # Heatmap
    if os.path.exists(IMAGES["heatmap"]):
        img = Image(IMAGES["heatmap"], width=7*inch, height=3.5*inch)
        story.append(img)
        story.append(Paragraph(
            "Figure 2: Sector performance conditional on each detected regime. The heatmap shows that the model found economically meaningful states— "
            "Energy and Materials performed best during Reflation, while Tech and Communication Services led during Risk-On regimes.",
            caption_style
        ))
        story.append(Spacer(1, 0.15*inch))

    # Timeline
    if os.path.exists(IMAGES["timeline"]):
        img = Image(IMAGES["timeline"], width=7*inch, height=4*inch)
        story.append(img)
        story.append(Paragraph(
            "Figure 3: Regime timeline annotated with major macroeconomic events. The model detected regime shifts around COVID, "
            "the 2022 rate hikes, and the SVB collapse without being provided these labels, "
            "indicating that the unsupervised model identified regime shifts that align with known macroeconomic events.",
            caption_style
        ))
        story.append(Spacer(1, 0.15*inch))

    # Drawdowns
    if os.path.exists(IMAGES["drawdown"]):
        img = Image(IMAGES["drawdown"], width=7*inch, height=3.5*inch)
        story.append(img)
        story.append(Paragraph(
            "Figure 4: Drawdown comparison over the out-of-sample period. The regime rotation strategies maintained shallower drawdowns "
            "than the benchmarks during this window.",
            caption_style
        ))
        story.append(Spacer(1, 0.15*inch))

    # Rolling Sharpe
    if os.path.exists(IMAGES["rolling_sharpe"]):
        img = Image(IMAGES["rolling_sharpe"], width=7*inch, height=3.5*inch)
        story.append(img)
        story.append(Paragraph(
            "Figure 5: Rolling 3-year Sharpe ratio. The HMM strategy maintained a positive Sharpe ratio throughout the evaluation period, "
            "though the ratio fluctuates, reflecting the time-dependent nature of strategy performance.",
            caption_style
        ))
        story.append(Spacer(1, 0.15*inch))

    # --- Section 7: Limitations ---
    story.append(Paragraph("VI. Limitations", heading_style))
    story.append(Paragraph(
        "Several limitations should be considered when interpreting these results:",
        body_style
    ))
    
    limitations_list = ListFlowable([
        ListItem(Paragraph("<b>Short out-of-sample period:</b> The evaluation window spans only 29 months (Aug 2023 – Dec 2025). This period was structurally unique due to AI-driven market concentration. Performance may not generalize to other environments.", body_style)),
        ListItem(Paragraph("<b>Look-ahead risk:</b> While feature engineering and walk-forward validation are designed to prevent look-ahead bias, the model's hyperparameters (regime count, transition prior) were selected using the full dataset. A separate hold-out period for hyperparameter selection would be needed in a production setting.", body_style)),
        ListItem(Paragraph("<b>Simplified transaction costs:</b> A fixed 5 bps cost per trade is applied, assuming mid-price execution with zero market impact—appropriate for a small notional portfolio. Institutional implementation would face variable spreads, market impact, and liquidity constraints not captured here.", body_style)),
        ListItem(Paragraph("<b>Regime stability:</b> The assumption that sectors behave consistently within a regime may break down over longer time horizons. The economic meaning of 'Technology' in 2025 differs from 2005.", body_style)),
        ListItem(Paragraph("<b>Always fully invested:</b> The strategy does not go to cash or hedge downside risk. This is a deliberate scoping choice to isolate the rotation signal. A natural extension would be a volatility-targeted overlay that reduces exposure during high-VIX regimes.", body_style)),
        ListItem(Paragraph("<b>US-centric:</b> The analysis uses US sector ETFs and macro data. Regime dynamics in other markets may differ.", body_style)),
    ], bulletType='bullet')
    story.append(limitations_list)
    story.append(Spacer(1, 0.2*inch))

    # --- Section 8: Summary ---
    story.append(Paragraph("VII. Summary", heading_style))
    story.append(Paragraph(
        "This project applies unsupervised learning to tactical asset allocation using a Hidden Markov Model for regime detection. "
        "The approach is transparent and interpretable: macro features are used to infer a latent market regime, "
        "and sector selection is conditioned on that regime based on historical performance patterns.",
        body_style
    ))
    story.append(Paragraph(
        "Over a 29-month out-of-sample period (Aug 2023 – Dec 2025), the HMM strategy achieved a Sharpe ratio of 1.64 with a maximum drawdown of -6.68%, "
        "compared to 0.72 and -23.93% for the S&P 500. The GMM baseline achieved a higher raw return (26.29% vs. 25.53%) but with higher volatility "
        "and deeper drawdowns, suggesting that the HMM's temporal modeling was associated with improved risk-adjusted performance in this sample.",
        body_style
    ))
    story.append(Paragraph(
        "These results are encouraging but come with important caveats: the out-of-sample period is short, the market environment was unusual, "
        "and transaction cost assumptions are simplified. The project is best viewed as a framework for regime-aware allocation rather than "
        "a complete strategy. The value lies in the problem framing, the deliberate modeling choices, and the honest treatment of limitations.",
        body_style
    ))
    story.append(Spacer(1, 0.2*inch))

    # --- Section 9: References ---
    story.append(Paragraph("VIII. References", heading_style))
    story.append(Paragraph(
        "1. Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency. <i>Journal of Finance</i>.<br/>"
        "2. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. <i>Proceedings of the IEEE</i>.<br/>"
        "3. Source Code: <a href='https://github.com/kira-ml/macro-regime-rotation'>https://github.com/kira-ml/macro-regime-rotation</a>",
        body_style
    ))

    # Build the PDF
    print("Building PDF document...")
    doc.build(story)
    print(f"✅ PDF successfully generated: {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_pdf()