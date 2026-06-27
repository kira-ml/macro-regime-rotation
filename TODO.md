# TODO.md - Macro-Informed Sector Rotation Project

## Project Status: **Final Production Release ✅**

---

## Completed Modules

### 1. Project Setup & Configuration ✅
- [x] Project directory structure created
- [x] `config.py` - Central configuration with all parameters
  - [x] Date range: 2006-01-01 to 2025-12-31
  - [x] 11 GICS sector ETFs configured
  - [x] 10 macro proxies from yfinance
  - [x] Model parameters (3 regimes, random seed 42)
  - [x] Strategy parameters (5 bps cost, top 3 sectors)
  - [x] Walk-forward settings (5-year initial training, annual refit)
  - [x] Feature names defined (9 features)
  - [x] Path management for data/output directories
- [x] `requirements.txt` - All dependencies with versions
  - [x] Core: pandas, numpy, scipy, scikit-learn
  - [x] ML: hmmlearn
  - [x] Data: yfinance
  - [x] Visualization: plotly, matplotlib, seaborn, kaleido
  - [x] Utilities: tqdm, pyarrow, joblib
  - [x] PDF: reportlab

### 2. Data Pipeline (`data.py`) ✅
- [x] `fetch_data()` - yfinance data acquisition
  - [x] Supports 22 tickers (11 sectors + 10 macro proxies + SPY)
  - [x] Progress bar integration
  - [x] Handles MultiIndex extraction
- [x] `resample_to_monthly()` - Frequency conversion
  - [x] Daily → Month-end resampling
  - [x] Handles missing data
- [x] `create_dynamic_universe()` - Sector availability tracking
  - [x] Handles XLRE (Oct 2015) and XLC (Jun 2018) inception dates
  - [x] Creates boolean mask of available sectors
- [x] `engineer_features()` - Feature engineering
  - [x] 9 features engineered from macro proxies:
    - [x] yield_curve_slope (10Y - 5Y)
    - [x] credit_spread_proxy (HYG/LQD)
    - [x] breakeven_proxy (TIP/IEF)
    - [x] vix_level (log transformed)
    - [x] vix_change (1-month)
    - [x] short_rate (T-Bill)
    - [x] gold_momentum_3m
    - [x] oil_momentum_3m
    - [x] credit_momentum_3m
- [x] `standardize_features()` - No look-ahead bias
  - [x] Rolling standardization
  - [x] Handles training/test splits
- [x] `get_walk_forward_splits()` - Train/test date generation
  - [x] Configurable min training years
  - [x] Configurable refit frequency
- [x] **Pipeline Hardening (Look-Ahead Bias Fix)**
  - [x] Added automatic 1-month feature shift (`features.shift(1)`) to ensure zero look-ahead bias
  - [x] Added auto-refresh logic for stale data (>30 days old)
  - [x] Explicitly drop NaNs before standardization
- [x] `get_regime_data()` - Main entry point
  - [x] Orchestrates complete data pipeline
  - [x] Saves to parquet (with CSV fallback)
  - [x] Returns features, sector_returns, available_mask
- [x] Data persistence
  - [x] Parquet file format (fast, compressed)
  - [x] CSV fallback if pyarrow unavailable

### 3. Models (`models.py`) ✅
- [x] **Baseline Model: GMM**
  - [x] `fit_gmm()` - Train Gaussian Mixture Model
  - [x] Regime distribution tracking
  - [x] Probability outputs
- [x] **Advanced Model: HMM**
  - [x] `fit_hmm()` - Train Hidden Markov Model
  - [x] Gaussian emissions with full covariance
  - [x] **Tuning:** Added `transmat_prior=10.0` to enforce state persistence and reduce flickering
  - [x] Transition matrix tracking
  - [x] Viterbi algorithm for regime prediction
- [x] `predict_regime()` - Unified prediction interface
  - [x] Supports both GMM and HMM
  - [x] Returns regimes and probabilities
- [x] `walk_forward_predictions()` - Walk-forward validation
  - [x] Expanding window (preserves rare regimes)
  - [x] Annual refit
  - [x] Progress bar tracking
  - [x] Stores all trained models
- [x] `characterize_regimes()` - Regime characterization
  - [x] Feature means by regime
  - [x] Automatic labeling heuristic
  - [x] Risk-On, Risk-Off, Reflation identification
- [x] `sector_performance_by_regime()` - Performance analysis
  - [x] Average returns by regime and sector
  - [x] Top sectors identification
- [x] `compare_models()` - GMM vs HMM comparison
  - [x] Agreement rate
  - [x] Transition counts (smoothness comparison)
  - [x] Top-3 sector overlap

### 4. Backtest (`backtest.py`) ✅
- [x] **Strategy Implementations**
  - [x] `run_regime_strategy()` - HMM/GMM rotation
    - [x] Sector selection by regime
    - [x] Equal weighting of selected sectors
    - [x] Dynamic sector universe handling
    - [x] Turnover tracking
  - [x] `run_equal_weight_benchmark()` - Benchmark 1
  - [x] `run_momentum_benchmark()` - Benchmark 2
    - [x] 6-month lookback (Jegadeesh & Titman)
    - [x] 1-month skip to avoid reversal
  - [x] **Added:** `run_spy_benchmark()` - S&P 500 Buy & Hold benchmark
  - [x] `run_full_backtest()` - Orchestration
- [x] **Transaction Costs**
  - [x] `apply_transaction_costs()` - 5 bps one-way
  - [x] Cost impact on returns
- [x] **Performance Metrics** (Tier 1)
  - [x] `calculate_metrics()` - Core metrics
    - [x] Annualized Return
    - [x] Annualized Volatility
    - [x] Sharpe Ratio
    - [x] Maximum Drawdown
    - [x] Calmar Ratio
    - [x] Win Rate
    - [x] Monthly Turnover
- [x] **Comparison**
  - [x] `compare_strategies()` - Table generation
  - [x] `calculate_cumulative_returns()`
- [x] **Visualizations**
  - [x] `plot_cumulative_returns()`
  - [x] `plot_regime_timeline()` (Smoothed using 3-month rolling average)
  - [x] `plot_drawdowns()`
  - [x] `plot_sector_heatmap()`

### 5. Evaluation (`evaluation.py`) ✅
- [x] **Tier 1: Core Metrics**
  - [x] `compute_metrics()` - Annual Return, Volatility, Sharpe, Max DD, Calmar, Win Rate
- [x] **Tier 2: Differentiating Metrics**
  - [x] `compute_regime_conditional_sharpe()` - Tests if regime model adds value
  - [x] `compute_rolling_sharpe()` - 3-year rolling window
  - [x] `compute_turnover()` - Implementation friction
- [x] **Comparison Table**
  - [x] `create_comparison_table()` - Single table, all strategies
- [x] **Quant-Finance Visualizations (Matplotlib/Seaborn)**
  - [x] `plot_hero_chart()` - Cumulative returns with regime background (White-grid, high-contrast colors)
  - [x] `plot_regime_heatmap()` - Sector performance by regime (RdYlGn colormap)
  - [x] `plot_drawdown_comparison()` - Underwater plot with shaded fills
  - [x] `plot_rolling_sharpe()` - Performance stability
  - [x] `plot_regime_timeline_annotated()` - Event validation with white-background text boxes
- [x] **Defensive Programming**
  - [x] Added try/except blocks to ensure plot generation doesn't crash the pipeline
- [x] **Output Generation**
  - [x] PNG exports for GitHub/LinkedIn (300 DPI, high resolution)

### 6. Documentation & Reporting ✅
- [x] **`experiment_regimes.py`** - Regime Count Validation
  - [x] Tested 2, 3, and 4 regimes for both GMM and HMM
  - [x] Confirmed 3 regimes is the optimal sweet spot for HMM
  - [x] Results table printed to console
- [x] **`generate_pdf_report.py`** - Academic PDF Generator
  - [x] Professional academic-style PDF (Times New Roman, justified)
  - [x] Contains Abstract, Methodology, Results, Limitations
  - [x] Embeds all 5 visualizations
  - [x] 6-page professional report
- [x] **`README.md`**
  - [x] Updated with final 3-regime metrics table
  - [x] Embedded all 5 visualizations directly into the README
  - [x] Added PDF download badge
  - [x] Updated methodology and feature engineering sections
- [x] **`.gitignore`**
  - [x] Fixed to **track PNG files** in `outputs/` so images load on GitHub
  - [x] Continued ignoring raw data files and HTML outputs

---

## Project Statistics

### Data
- **Date Range:** 2018-08-31 to 2025-12-31 (89 months)
- **Sectors:** 11 (all GICS)
- **Macro Proxies:** 10 + SPY (Benchmark)
- **Features:** 9 engineered

### Models
- **GMM:** 3 regimes, walk-forward validated
- **HMM:** 3 regimes, walk-forward validated
- **Model tuning:** `transmat_prior=10.0` applied for persistence

### Performance (Final HMM Strategy - 3 Regimes)
| Metric | Value |
|--------|-------|
| Annual Return | **25.53%** |
| Volatility | **12.69%** |
| Sharpe Ratio | **1.85** |
| Max Drawdown | **-6.68%** |
| Calmar Ratio | **381.98%** |
| Win Rate | 72.41% |
| Monthly Turnover | 16.09% |

### Benchmark Comparison (Out-of-Sample 2023–2025)
| Metric | HMM (3 States) | GMM | SPY | Momentum |
|--------|----------------|-----|-----|----------|
| Sharpe Ratio | **1.85** | 1.71 | 0.74 | 0.71 |
| Max Drawdown | **-6.68%** | -10.23% | -23.93% | -16.61% |

### Files Generated
- **Data:** 5 parquet files (sector_prices, macro_prices, features, sector_returns, available_mask)
- **Visualizations:** 5 high-res PNG files (300 DPI)
- **Code:** 7 Python modules (~1,800+ lines)
- **Documentation:** 1 PDF report, 1 README, 1 TODO.md

---

## Known Limitations

1. **Out-of-Sample Period:** Only 29 out-of-sample months (2023–2025)
   - Limited statistical significance
   - Period was characterized by strong AI-driven concentration

2. **Regime 2 (Reflation):** Rare event (1 month only)
   - Model correctly identified it as rare, but cannot be robustly evaluated

3. **Transaction Costs:** Simple 5 bps assumed for all trades
   - Real execution may have higher costs for less liquid ETFs

4. **No Risk Management:** Always 100% invested
   - No cash position or volatility targeting

5. **Feature Set:** Limited to 9 macro proxies from yfinance
   - Excludes geopolitical risk, sentiment, or central bank communication

---

## Next Steps / Future Work

### Short-Term Enhancements
- [ ] **Dynamic Risk-Free Rate:** Replace fixed 3% with actual 3-month T-bill (^IRX)
- [ ] **Risk-Managed Overlay:** Add cash hedge or volatility targeting during Regime 2
- [ ] **Feature Expansion:** Add Fed speeches or geopolitical risk indices
- [ ] **Transformer Comparison:** Test transformer-based architectures for continuous regime probability

### Medium-Term Improvements
- [ ] **International Expansion:** Test the strategy on European or Asian sector ETFs
- [ ] **Streamlit Dashboard:** Build an interactive dashboard for live regime tracking
- [ ] **Ensemble Averaging:** Combine GMM and HMM predictions for smoother transitions

### Long-Term Research
- [ ] **Deep Learning:** Replace HMM with LSTM/Transformers for sequential regime detection
- [ ] **Real-Time Deployment:** Build a pipeline that pulls data daily and updates signals

---

## How to Run the Complete Project

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Run full pipeline (Data → Models → Backtest → Evaluation)
python evaluation.py

# 3. Run the Regime Count Experiment (2, 3, 4 states)
python experiment_regimes.py

# 4. Generate the Academic PDF Report
python generate_pdf_report.py
```

---

## Project Structure

```
macro-regime-rotation/
├── config.py              # Configuration (✅ Complete)
├── data.py               # Data pipeline (✅ Complete)
├── models.py             # GMM + HMM (✅ Complete)
├── backtest.py          # Strategy backtest (✅ Complete)
├── evaluation.py        # Metrics + visualizations (✅ Complete)
├── experiment_regimes.py # Regime count experiment (✅ Complete)
├── generate_pdf_report.py # PDF generator (✅ Complete)
├── requirements.txt     # Dependencies (✅ Complete)
├── README.md            # Project documentation (✅ Complete)
├── TODO.md              # This file (✅ Complete)
├── Macro_Regime_Rotation_Report.pdf # Academic report (✅ Complete)
│
├── data/               # Processed data (✅ Generated)
│   ├── sector_prices.parquet
│   ├── macro_prices.parquet
│   ├── features.parquet
│   ├── sector_returns.parquet
│   └── available_mask.parquet
│
└── outputs/            # Visualizations (✅ Generated)
    ├── hero_chart.png
    ├── regime_heatmap.png
    ├── drawdowns.png
    ├── rolling_sharpe.png
    └── regime_timeline_events.png
```

---

## Key Achievements

1. **Zero Look-Ahead Bias** - Features shifted 1-month backward to ensure no future data leaks
2. **Walk-Forward Validation** - 5-year initial training, annual refits, expanding window
3. **Transaction Costs** - Realistic 5 bps included in all backtests
4. **Dynamic Universe** - Handles XLRE (2015) and XLC (2018) inception dates
5. **HMM Persistence Tuning** - `transmat_prior=10.0` eliminated flickering (turnover dropped from 62% to 16%)
6. **Empirical Regime Validation** - Tested 2, 3, and 4 regimes; confirmed 3 is optimal
7. **Professional Visualizations** - 5 publication-grade Matplotlib/Seaborn charts
8. **Academic Documentation** - 6-page PDF report + fully documented README

---

**Last Updated:** 2026-06-27
**Status:** Final Production Release ✅
