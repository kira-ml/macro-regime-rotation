# TODO.md - Macro-Informed Sector Rotation Project

## Project Status: Foundation MVP Complete ✅

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
  - [x] Utilities: tqdm, pyarrow

### 2. Data Pipeline (`data.py`) ✅
- [x] `fetch_data()` - yfinance data acquisition
  - [x] Supports 21 tickers (11 sectors + 10 macro proxies)
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
  - [x] `plot_regime_timeline()`
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
- [x] **LinkedIn-Ready Visualizations**
  - [x] `plot_hero_chart()` - Cumulative returns with regime background
  - [x] `plot_regime_heatmap()` - Sector performance by regime
  - [x] `plot_drawdown_comparison()` - Underwater plot
  - [x] `plot_rolling_sharpe()` - Performance stability
  - [x] `plot_regime_timeline_annotated()` - Event validation
- [x] **Output Generation**
  - [x] HTML exports for interactive charts
  - [x] PNG exports for LinkedIn posts
  - [x] High-resolution (scale=2)

---

## Project Statistics

### Data
- **Date Range:** 2018-07-31 to 2025-12-31 (90 months)
- **Sectors:** 11 (all GICS)
- **Macro Proxies:** 10
- **Features:** 9 engineered

### Models
- **GMM:** 3 regimes, walk-forward validated
- **HMM:** 3 regimes, walk-forward validated
- **Models trained:** 3 per approach (annual refit)

### Performance (HMM Strategy)
| Metric | Value |
|--------|-------|
| Annual Return | 18.71% |
| Volatility | 12.63% |
| Sharpe Ratio | 1.19 |
| Max Drawdown | -7.29% |
| Calmar Ratio | 2.57 |
| Win Rate | 63.33% |
| Monthly Turnover | 62.22% |

### Files Generated
- **Data:** 5 parquet files (sector_prices, macro_prices, features, sector_returns, available_mask)
- **Visualizations:** 5 HTML + 5 PNG files
- **Code:** 5 Python modules (~1,500+ lines)

---

## Known Limitations

1. **Data Range:** Due to XLC inception (2018), only 90 months of data available
   - Trade-off: All 11 sectors available vs. longer history
   - Consider: Dropping XLC for 2006-2025 history

2. **Regime Labels:** Heuristic labeling may not perfectly match economic intuition
   - Current labels: Mixed, Reflation/Inflationary, Mixed
   - Opportunity: Manual labeling with economic validation

3. **Sample Size:** Only 30 out-of-sample months for walk-forward
   - Limited statistical significance
   - Need more data or longer training period

4. **Transaction Costs:** Simple 5 bps assumed for all trades
   - Real execution may have higher costs for less liquid ETFs

5. **No Risk Management:** Always 100% invested
   - No cash position or volatility targeting

---

## Next Steps / Future Work

### Short-Term Enhancements
- [ ] Add dynamic regime labeling with economic indicators
- [ ] Include SPY as additional benchmark
- [ ] Add volatility targeting overlay
- [ ] Create interactive dashboard with Streamlit
- [ ] Add sensitivity analysis for number of regimes (2, 3, 4)

### Medium-Term Improvements
- [ ] Expand feature set with more macro indicators
- [ ] Add regime persistence constraints
- [ ] Implement ensemble of GMM/HMM with averaging
- [ ] Add regime probability thresholds (only trade with high confidence)
- [ ] Include regime transition penalties

### Long-Term Research
- [ ] Compare with deep learning approaches (LSTM, Transformer)
- [ ] Test on international markets
- [ ] Add alternative data (sentiment, options flow)
- [ ] Build real-time deployment pipeline

---

## How to Run the Complete Project

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Data pipeline
python data.py

# 3. Train models
python models.py

# 4. Run backtest
python backtest.py

# 5. Generate evaluation & visualizations
python evaluation.py
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
├── requirements.txt     # Dependencies (✅ Complete)
├── TODO.md             # This file
│
├── data/               # Processed data (✅ Generated)
│   ├── sector_prices.parquet
│   ├── macro_prices.parquet
│   ├── features.parquet
│   ├── sector_returns.parquet
│   └── available_mask.parquet
│
└── outputs/            # Results (✅ Generated)
    ├── hero_chart.html/png
    ├── regime_heatmap.html/png
    ├── drawdowns.html/png
    ├── rolling_sharpe.html/png
    └── regime_timeline_events.html/png
```

---

## Key Achievements

1. **Zero Look-Ahead Bias** - All features use data available at prediction time
2. **Walk-Forward Validation** - Models retrained annually, tested out-of-sample
3. **Transaction Costs** - Realistic 5 bps included
4. **Dynamic Universe** - Handles ETFs with different inception dates
5. **Reproducible** - Single source of truth with config.py
6. **Complete Story** - From economic theory → features → model → strategy → evaluation
7. **LinkedIn-Ready** - Professional visualizations for portfolio presentation

---

**Last Updated:** 2026-06-27
**Status:** Foundation MVP Complete ✅