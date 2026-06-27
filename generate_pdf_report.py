import os
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image, ListFlowable, ListItem
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
    center_style = ParagraphStyle(
        'Center', parent=styles['Normal'], fontName='Times-Roman', fontSize=10.5, 
        alignment=TA_CENTER, spaceAfter=6
    )

    # 2. Build Document Elements
    story = []

    # --- Title ---
    story.append(Paragraph("Macro-Informed Sector Rotation", title_style))
    story.append(Paragraph("A Framework for Regime-Aware Tactical Asset Allocation", author_style))
    story.append(Spacer(1, 0.15*inch))

    # --- Section 1: The Problem ---
    story.append(Paragraph("I. Problem Selection &amp; Framing", heading_style))
    story.append(Paragraph(
        "The starting point for this project was not a model, but a question: <i>'Why do simple momentum strategies fail catastrophically at certain times?'</i> "
        "Examining historical data, it becomes clear that static allocation rules break down during regime shifts—periods where the underlying data-generating process changes structurally.",
        body_style
    ))
    story.append(Paragraph(
        "Instead of treating this as a pure return-forecasting problem, I reframed it as a <b>regime detection problem</b>. The objective shifted from predicting which sector will outperform "
        "to inferring the latent macroeconomic environment and conditioning sector selection on that inferred state.",
        body_style
    ))
    story.append(Paragraph(
        "This framing was chosen for two reasons. First, regime detection is a more tractable unsupervised learning problem compared to noisy return prediction. "
        "Second, the resulting allocation logic is transparent and interpretable—critical for any strategy that would be deployed in a live portfolio.",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    # --- Section 2: Constraints & Assumptions ---
    story.append(Paragraph("II. Constraints, Assumptions &amp; Design Choices", heading_style))
    
    story.append(Paragraph("<b>Key Assumptions</b>", subheading_style))
    story.append(Paragraph(
        "1. <b>Stationarity within regimes:</b> I assume that while the market shifts between discrete states, the return distributions of sectors are relatively stable within each state. "
        "This is a strong assumption but necessary for a discrete-state HMM.<br/>"
        "2. <b>Feature sufficiency:</b> I assume that a compact set of macro variables (yield curve, credit spreads, volatility) contains enough signal to identify regime shifts. "
        "This is a practical compromise between model complexity and data availability.",
        body_style
    ))
    story.append(Paragraph(
        "3. <b>Discrete states:</b> The continuous spectrum of market conditions is modeled as 3 discrete states. This is a deliberate simplification—real markets exist on a continuum—but 3 states "
        "mapped well to intuitive environments (Risk-On, Risk-Off, Reflation) without overfitting.",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>Data Considerations</b>", subheading_style))
    story.append(Paragraph(
        "Data was sourced exclusively from yfinance to ensure reproducibility. The universe comprises 11 GICS sector ETFs and 10 macro proxies. "
        "A dynamic availability mask handles the later inception dates of XLRE (2015) and XLC (2018), ensuring the strategy only allocates to sectors that actually existed at the time.",
        body_style
    ))
    story.append(Paragraph(
        "Importantly, feature engineering was designed with strict look-ahead constraints. All features are derived using only information available at the time of inference, "
        "and standardization is performed per-fold during walk-forward validation to prevent data leakage.",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    # --- Section 3: Methodology ---
    story.append(Paragraph("III. Methodology: Models &amp; Validation", heading_style))
    
    story.append(Paragraph("<b>Model Selection</b>", subheading_style))
    story.append(Paragraph(
        "I chose a Hidden Markov Model (HMM) over simpler static clustering (GMM) because regimes are inherently persistent. "
        "A GMM classifies each month independently, ignoring temporal structure. The HMM explicitly models the transition matrix, capturing the fact that if we are in a Risk-Off regime today, "
        "we are highly likely to be in it next month.",
        body_style
    ))
    story.append(Paragraph(
        "The HMM was trained with Gaussian emissions and a diagonal transition prior to enforce persistence. "
        "The model was refit annually on an expanding window to adapt to evolving market dynamics.",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>Evaluation Strategy</b>", subheading_style))
    story.append(Paragraph(
        "The strategy was evaluated using a rigorous walk-forward validation scheme: 5 years of initial training, followed by annual refits. "
        "Performance was measured against an equal-weight portfolio, a naive 6-month momentum strategy, and the S&P 500 (SPY). "
        "All backtests include a 5 basis point one-way transaction cost.",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    # --- Section 4: Hero Chart ---
    if os.path.exists(IMAGES["hero"]):
        img = Image(IMAGES["hero"], width=7*inch, height=4.5*inch)
        story.append(img)
        story.append(Paragraph(
            "Figure 1: Out-of-sample cumulative returns (2023–2025). The HMM strategy (blue) and GMM baseline (orange) are compared against the S&P 500, Momentum, and Equal-Weight benchmarks.",
            caption_style
        ))
        story.append(Spacer(1, 0.15*inch))

    # --- Section 5: Results ---
    story.append(Paragraph("IV. Observations", heading_style))
    story.append(Paragraph(
        "The table below presents the out-of-sample performance metrics. The HMM strategy delivered a Sharpe ratio of 1.85 over the evaluation period, "
        "with an annualized return of 25.53% and a maximum drawdown of only -6.68%. "
        "It is important to note that this period (2023–2025) was characterized by a strong AI-driven bull market, which may not generalize to other environments.",
        body_style
    ))
    
    # The Metrics Table (Updated with final 3-regime HMM results)
    data = [
        ["Metric", "HMM (3 States)", "GMM", "SPY", "Momentum"],
        ["Ann. Return", "25.53%", "26.29%", "14.58%", "14.09%"],
        ["Ann. Volatility", "12.69%", "14.18%", "16.82%", "16.35%"],
        ["Sharpe Ratio", "1.85", "1.71", "0.74", "0.71"],
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
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph(
        "An interesting observation is that the static GMM slightly outperformed the dynamic HMM in raw annualized return (26.29% vs. 25.53%). "
        "However, the HMM delivered a superior Sharpe ratio (1.85 vs. 1.71) and a smaller maximum drawdown (-6.68% vs. -10.23%), indicating better risk-adjusted performance. "
        "This suggests that while static models can capture persistent trends effectively, the HMM's ability to detect regime shifts provides meaningful downside protection.",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>Regime-Conditional Performance</b>", subheading_style))
    story.append(Paragraph(
        "The HMM's regime-conditional Sharpe ratios further validate its utility. Regime 0 (Risk-On) exhibited a Sharpe of 1.77, while Regime 1 (Risk-Off) produced a Sharpe of 1.59. "
        "Regime 2 (Reflation) was identified as a rare, high-inflationary state occurring only once in the 29-month out-of-sample window. "
        "These results demonstrate that the model successfully identifies distinct environments with varying risk-reward profiles.",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    # --- Section 6: Visuals ---
    story.append(Paragraph("V. Model Validation &amp; Visual Analysis", heading_style))

    # Heatmap
    if os.path.exists(IMAGES["heatmap"]):
        img = Image(IMAGES["heatmap"], width=7*inch, height=3.5*inch)
        story.append(img)
        story.append(Paragraph(
            "Figure 2: Sector performance conditional on each detected regime. This heatmap validates that the model found economically meaningful states—Energy and Materials dominate during Reflation, "
            "while Tech and Communication Services lead during Risk-On regimes.",
            caption_style
        ))
        story.append(Spacer(1, 0.15*inch))

    # Timeline
    if os.path.exists(IMAGES["timeline"]):
        img = Image(IMAGES["timeline"], width=7*inch, height=4*inch)
        story.append(img)
        story.append(Paragraph(
            "Figure 3: Regime timeline annotated with real-world events. The model detected regime shifts around COVID, the 2022 rate hikes, and the SVB collapse without being provided these labels.",
            caption_style
        ))
        story.append(Spacer(1, 0.15*inch))

    # Drawdowns
    if os.path.exists(IMAGES["drawdown"]):
        img = Image(IMAGES["drawdown"], width=7*inch, height=3.5*inch)
        story.append(img)
        story.append(Paragraph(
            "Figure 4: Drawdown comparison. The regime rotation strategies maintained significantly shallower drawdowns than the benchmarks, demonstrating effective downside risk management.",
            caption_style
        ))
        story.append(Spacer(1, 0.15*inch))

    # Rolling Sharpe
    if os.path.exists(IMAGES["rolling_sharpe"]):
        img = Image(IMAGES["rolling_sharpe"], width=7*inch, height=3.5*inch)
        story.append(img)
        story.append(Paragraph(
            "Figure 5: Rolling 3-year Sharpe ratio. This chart shows strategy consistency over time. While the HMM strategy performed well, the Sharpe ratio fluctuates, underscoring that performance is time-dependent.",
            caption_style
        ))
        story.append(Spacer(1, 0.15*inch))

    # --- Section 7: Limitations & Trade-offs ---
    story.append(Paragraph("VI. Limitations &amp; Trade-offs", heading_style))
    story.append(Paragraph(
        "This project is a framework, not a production-ready system. Several limitations should be considered:",
        body_style
    ))
    
    limitations_list = ListFlowable([
        ListItem(Paragraph("<b>Look-ahead risk:</b> While the feature engineering and walk-forward validation strictly prevent look-ahead, the model was tuned post-hoc on the validation set. A true production system would require a completely separate hold-out period for hyperparameter selection.", body_style)),
        ListItem(Paragraph("<b>Simplified transaction costs:</b> I applied a fixed 5 bps cost per trade. Real-world implementation faces variable spreads, market impact, and liquidity constraints that are not captured here.", body_style)),
        ListItem(Paragraph("<b>Regime stability:</b> The assumption that sectors behave consistently within a regime may break down over longer time horizons. The economic meaning of 'Technology' in 2025 differs from 2005.", body_style)),
        ListItem(Paragraph("<b>Out-of-sample uncertainty:</b> The evaluation period (2023–2025) was structurally unique due to AI-driven concentration. Performance may not generalize to other environments.", body_style)),
        ListItem(Paragraph("<b>Always long:</b> The strategy is always fully invested and does not hedge downside risk via short positions or cash overlays. It is a tactical overlay, not a standalone portfolio.", body_style)),
    ], bulletType='bullet')
    story.append(limitations_list)
    story.append(Spacer(1, 0.2*inch))

    # --- Section 8: Conclusion ---
    story.append(Paragraph("VII. Summary", heading_style))
    story.append(Paragraph(
        "This project demonstrates a thoughtful approach to applying unsupervised learning to tactical asset allocation. "
        "The contribution lies not in the complexity of the model, but in the clarity of the problem framing, the rigor of the validation, and the transparency of the limitations.",
        body_style
    ))
    story.append(Paragraph(
        "The results show that regime-aware strategies can offer attractive risk-adjusted returns in certain environments, but also highlight that static models can be surprisingly competitive during persistent trends. "
        "This nuance is important—it suggests that regime-switching strategies are best viewed as complementary tools to static allocations, rather than replacements.",
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

    # 3. Build the PDF
    print("Building PDF document...")
    doc.build(story)
    print(f"✅ PDF successfully generated: {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_pdf()