# Macro-Informed Sector Rotation: A Regime-Switching Approach to Tactical Asset Allocation

> **A Data Science project demonstrating macro-quantitative thinking, unsupervised regime detection, and rigorous strategy evaluation for tactical asset allocation.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

---

## 📌 Table of Contents

- [Problem Framing](#-problem-framing)
- [Why This Matters in Quantitative Investing](#-why-this-matters-in-quantitative-investing)
- [Project Objectives](#-project-objectives)
- [Assumptions and Limitations](#-assumptions-and-limitations)
- [Data Sources](#-data-sources)
- [Methodology](#-methodology)
- [Repository Structure](#-repository-structure)
- [Key Results](#-key-results)
- [Skills Demonstrated](#-skills-demonstrated)
- [Getting Started](#-getting-started)
- [License](#-license)

---

## 🧠 Problem Framing

### The Core Problem

Static asset allocation and naive momentum strategies share a critical vulnerability: **they are blind to the prevailing macroeconomic environment.** A strategy that excels during low-volatility, risk-on expansions can suffer catastrophic drawdowns when the regime abruptly shifts to a flight-to-safety recessionary panic.

This project reframes tactical asset allocation as a **regime detection problem** rather than a pure return-forecasting problem. Instead of asking *"Which sector will go up next month?"* — a notoriously difficult and noisy prediction task — we ask a more tractable question:

> **"Given the current macro environment, which sectors have historically performed best in similar regimes?"**

### The Analytical Approach

The intellectual foundation rests on three observations:

| Observation | Implication |
|-------------|-------------|
| Financial market data is **non-stationary** — return distributions, correlations, and factor behavior change through time. | Models that assume a single unchanging data-generating process will inevitably fail at turning points. |
| Macroeconomic variables (yield spreads, credit conditions, volatility) provide **leading or coincident signals** about the prevailing market regime. | We can infer the latent regime from observable data without needing to predict it with perfect foresight. |
| Sector performance conditional on regime exhibits **persistent, economically intuitive patterns** (e.g., Utilities outperform during recessionary risk-off; Technology leads during disinflationary growth). | A simple mapping from inferred regime → historical sector performance can form the basis of a defensible tactical allocation rule. |

### Modeling Philosophy

This project deliberately favors **interpretability over black-box complexity.** The goal is not to build the most sophisticated model possible, but to solve a well-defined problem with a transparent, auditable methodology. Every modeling choice is justified by the structure of the problem:

- **Unsupervised learning** is appropriate because true regime labels do not exist ex-ante. We cannot supervise on "recession" because recessions are declared months in retrospect.
- **Hidden Markov Models** explicitly capture temporal state persistence — the fact that if we are in a risk-off regime today, we are highly likely to remain in it tomorrow. This temporal structure is the key advantage over static clustering methods.
- **Post-hoc regime characterization** ensures the model output is economically interpretable and can be validated against known historical events.

### The Baseline-First Principle

Before deploying any advanced model, we establish performance benchmarks that serve dual purposes: they define the minimum hurdle any active strategy must clear, and they cleanly isolate the marginal value of each modeling decision.

1. **Equal-Weight Portfolio** — The simplest possible allocation. If a regime strategy cannot beat passive diversification, it adds no value.
2. **Naive Momentum Strategy** — A well-documented factor that captures cross-sectional trends without any macro awareness. Beating momentum demonstrates that macro information provides signal beyond price-based factors.
3. **Gaussian Mixture Model (GMM)** — A static clustering approach that classifies each period independently. Comparing GMM to HMM directly tests whether **modeling temporal sequence** adds explanatory power. If HMM does not outperform GMM, we have learned something important about the data.

This progression is not just methodological hygiene — it is a direct signal to hiring managers that you understand when and why to reach for complexity.

---

## 💼 Why This Matters in Quantitative Investing

Quantitative investing has evolved far beyond pure price-based factor models. The modern quant must integrate alternative data sources, and macroeconomic data is among the most impactful. This project demonstrates capabilities that are directly relevant to roles at multi-manager platforms, macro hedge funds, and asset allocation teams:

| Business Need | How This Project Addresses It |
|---------------|-------------------------------|
| **Managing downside risk during regime shifts** | The HMM provides a systematic, non-discretionary signal for reducing exposure to vulnerable sectors before drawdowns materialize. |
| **Bridging econometrics and machine learning** | Feature engineering translates economic theory (yield curve dynamics, credit spreads) into ML-compatible inputs without data leakage. |
| **Building interpretable systematic strategies** | Every step — from regime labeling to sector selection — can be explained to portfolio managers and risk officers. No black boxes. |
| **Demonstrating strategy robustness** | Walk-forward validation and transaction-cost-aware backtesting mirror the standards expected in real investment research. |
| **Adapting to non-stationary markets** | The project explicitly models the reality that relationships shift across regimes, rather than assuming a single unchanging world. |

---

## 🎯 Project Objectives

1. **Engineer Investable Features:** Design a compact set of non-redundant, economically motivated macro features (yield curve slope, credit spreads, volatility measures) that precede or define observable market regimes.

2. **Develop an Interpretable Regime Detection Model:** Fit a Hidden Markov Model to infer 3–4 discrete, latent market states using only information available at each point in time. Validate detected regimes against known macroeconomic history.

3. **Build a Transparent Decision Logic:** For each inferred regime, compute the average forward returns of major US equity sectors. Create a systematic rule: allocate capital to historically best-performing sectors when a given regime is detected.

4. **Conduct a Realistic Strategy Backtest:** Implement a monthly sector rotation strategy. Evaluate against equal-weight and momentum benchmarks with explicit transaction cost assumptions. Report standard industry metrics (Sharpe ratio, maximum drawdown, Calmar ratio).

5. **Visually Narrate Findings:** Produce publication-quality visualizations mapping the regime timeline to cumulative strategy performance, annotated with major macroeconomic events for intuitive validation.

---

## ⚠️ Assumptions and Limitations

*A mature data scientist identifies what their model cannot do as clearly as what it can. This section is a feature of the project, not an afterthought.*

### Assumptions

| Assumption | Justification | Risk |
|------------|---------------|------|
| **Stationarity within regimes** | Sector return characteristics are reasonably stable within a given macro state, even if they differ dramatically across states. | If within-regime dynamics are highly unstable, the conditional performance mapping degrades. |
| **3–4 discrete regimes capture the essential dynamics** | A small number of states maps cleanly to intuitive market environments (Risk-On, Risk-Off, Reflation, Stagflation) and avoids overfitting. | Continuous market nuances are lost. The model cannot express "somewhat risk-off." |
| **Regime-sector relationships are persistent** | Historically observed patterns (e.g., defensives outperform in recessions) will broadly persist due to structural economic mechanisms. | Sector composition changes over decades. "Technology" today differs fundamentally from "Technology" in 2005. |
| **Macro features are sufficient for regime identification** | A curated set of 5–7 macro variables captures the key drivers of regime shifts. | Excluded variables (geopolitical risk, central bank communication, sentiment) may contain incremental signal. |

### Limitations (Honestly Stated)

- **No Out-of-Sample Certainty:** Walk-forward validation provides realistic performance estimates, but all financial backtests are fundamentally limited by the available historical sample. Regimes we have not yet observed cannot be modeled.
- **Look-Ahead Bias Risk:** The most critical technical challenge is ensuring zero information leakage. Every feature at time `t` must be constructed using only data available at or before `t`. This is rigorously enforced but requires meticulous implementation.
- **Simplified Transaction Costs:** A fixed basis-point cost per trade is applied. Real-world implementation faces variable spreads, market impact, and capacity constraints not modeled here.
- **Always Fully Invested:** The strategy rotates between sectors but never goes to cash or short. It is a relative allocation tool, not a complete risk management system. A real portfolio would overlay position sizing and stop-loss rules.
- **No Performance Attribution Decomposition:** The backtest does not decompose returns into regime-timing skill versus sector-selection skill versus structural beta exposure. This would require a more formal attribution framework.
- **US-Centric Analysis:** Sector ETFs and macro data are US-focused. Regime dynamics in other markets may differ.

---

## 📊 Data Sources

| Source | Data | Frequency | Period |
|--------|------|-----------|--------|
| **yfinance** | Sector ETFs (XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY) | Daily | ~2006–Present |
| **FRED** (via `pandas-datareader`) | 10Y-2Y Treasury spread, VIX, CPI, Industrial Production, Unemployment Claims, Fed Funds Rate | Daily/Monthly | ~2006–Present |
| **FRED** | Corporate credit spreads (BAA-AAA), High Yield OAS | Daily | ~2006–Present |

> **Note:** Start date is constrained by the inception of the newest sector ETF in the universe.

---

## 🔬 Methodology

### Feature Engineering

Raw data is transformed into economically meaningful, non-redundant features:

| Feature | Economic Interpretation | Signal Type |
|---------|------------------------|-------------|
| **Yield Curve Slope (10Y-2Y)** | Growth expectations, recession probability | Leading |
| **Credit Spread (BAA-AAA)** | Corporate distress risk, risk appetite | Coincident |
| **VIX Level (30-day moving average)** | Equity market fear gauge | Coincident |
| **VIX Change (1-month momentum)** | Shifts in uncertainty regime | Transitional |
| **Sector Momentum Dispersion** | Cross-sectional disagreement, regime instability | Derived |

All features are standardized using an expanding window to prevent look-ahead bias.

### Modeling Progression

```
┌─────────────────────────────────────────────┐
│            BASELINE SOLUTIONS               │
├─────────────────────────────────────────────┤
│  1. Equal-Weight Portfolio                  │
│  2. Naive Momentum (Top-3, 6-month lookback)│
│  3. Gaussian Mixture Model (static clusters)│
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│           ADVANCED SOLUTION                  │
├─────────────────────────────────────────────┤
│  4. Hidden Markov Model (temporal dynamics) │
│     - Gaussian emissions                     │
│     - 3-4 hidden states                      │
│     - Walk-forward regime inference          │
└─────────────────────────────────────────────┘
```

### Strategy Logic

1. **At each month-end:** Using only past data, compute the feature vector.
2. **Regime inference:** The trained HMM outputs the probability of being in each regime state.
3. **Regime assignment:** Select the highest-probability regime as the current state.
4. **Sector selection:** Allocate equally to the top-3 sectors with the highest average forward return historically conditioned on this regime.
5. **Rebalance:** Execute trades at next open. Subtract transaction costs.
6. **Repeat:** Expand the training window and re-estimate the model annually.

### Evaluation Framework

- **Walk-Forward Validation:** Models are never trained on future data. Annual refitting with expanding window.
- **Performance Metrics:** Annualized return, annualized volatility, Sharpe ratio, maximum drawdown, Calmar ratio, win rate.
- **Benchmark Comparison:** Direct head-to-head against all three baselines.

---

## 📁 Repository Structure

```
macro-regime-rotation/
│
├── README.md                          # Project documentation (you are here)
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # ETF and macro data pulling
│   ├── 02_feature_engineering.ipynb   # Feature creation and EDA
│   ├── 03_regime_detection.ipynb      # GMM and HMM model development
│   └── 04_strategy_backtest.ipynb     # Rotation strategy and evaluation
│
├── src/
│   ├── data.py                        # Data loading utilities
│   ├── features.py                    # Feature engineering functions
│   ├── models.py                      # GMM and HMM model classes
│   ├── backtest.py                    # Strategy backtest engine
│   └── viz.py                         # Visualization helpers
│
├── data/                              # Processed data (CSV format)
│   ├── sector_prices.csv
│   ├── macro_features.csv
│   └── regime_labels.csv
│
└── outputs/
    ├── regime_timeline.html           # Interactive regime visualization
    ├── performance_summary.csv        # Strategy vs benchmark metrics
    └── figures/                       # Static PNG exports
```

---

## 📈 Key Results

> *[To be populated after project execution. Below is a template.]*

| Metric | Equal-Weight | Momentum | GMM Rotation | HMM Rotation |
|--------|--------------|----------|--------------|--------------|
| Annualized Return | — | — | — | — |
| Annualized Volatility | — | — | — | — |
| Sharpe Ratio | — | — | — | — |
| Max Drawdown | — | — | — | — |
| Calmar Ratio | — | — | — | — |

### Regime Characterization

| Regime | Label | Characteristic Conditions | Best Sectors | Worst Sectors |
|--------|-------|---------------------------|--------------|---------------|
| State 0 | — | — | — | — |
| State 1 | — | — | — | — |
| State 2 | — | — | — | — |
| State 3 | — | — | — | — |

---

## 💡 Skills Demonstrated to Hiring Managers

| Skill Category | Specific Demonstration |
|----------------|------------------------|
| **Quantitative Finance** | Tactical asset allocation, factor timing, transaction cost modeling, benchmark selection, performance evaluation. |
| **Feature Engineering** | Translating economic theory (yield curve, credit spreads, vol dynamics) into investable, leakage-free features. |
| **Probabilistic Machine Learning** | Practical application of HMMs and GMMs for time-series problems with clear justification for model selection. |
| **Model Validation** | Walk-forward validation, baseline comparison methodology, post-hoc economic validation of unsupervised states. |
| **Scientific Communication** | Explicit assumptions and limitations, clean repository structure, reproducible workflow, narrative-driven README. |

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.9+
```

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/macro-regime-rotation.git
cd macro-regime-rotation
pip install -r requirements.txt
```

### Running the Analysis

Execute notebooks in order:

```bash
jupyter notebook notebooks/01_data_acquisition.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_regime_detection.ipynb
jupyter notebook notebooks/04_strategy_backtest.ipynb
```

Or run the full pipeline:

```bash
python src/main.py
```

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Contact

Built as a portfolio project demonstrating data science for quantitative investing. Questions or feedback are welcome via GitHub Issues.

---

> *"The value of a model lies not in its complexity, but in the clarity of the problem it solves and the honesty with which its limitations are communicated."*
