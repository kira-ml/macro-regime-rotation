# config.py
"""
Central configuration for macro-regime-rotation project.
All data sourced from yfinance for maximum reproducibility.
All parameters are documented with rationale.
"""

from pathlib import Path
import os

# ============================================================================
# PATHS
# ============================================================================
# Use Path objects for cross-platform compatibility
PROJECT_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATE RANGE
# ============================================================================
START_DATE = "2006-01-01"
END_DATE = "2025-12-31"


# ============================================================================
# SECTOR ETFs (All 11 GICS sectors)
# ============================================================================
SECTOR_ETFS = {
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLU": "Utilities",
    "XLRE": "Real Estate",      # Launched Oct 2015
    "XLC": "Communication Services",  # Launched Jun 2018
    "XLB": "Materials",
}


# ============================================================================
# MACRO PROXIES (All from yfinance — no FRED API key required)
# ============================================================================
MACRO_PROXIES = {
    "^VIX": "CBOE Volatility Index",
    "^TNX": "10-Year Treasury Yield",
    "^FVX": "5-Year Treasury Yield",
    "^IRX": "13-Week T-Bill Rate (Short Rate Proxy)",
    "LQD":  "Investment Grade Corporate Bond ETF",
    "HYG":  "High Yield Corporate Bond ETF",
    "TIP":  "TIPS Bond ETF",
    "IEF":  "7-10 Year Treasury Bond ETF",
    "GLD":  "Gold ETF (Safe Haven Proxy)",
    "USO":  "Oil ETF (Commodity Cycle Proxy)",
}

# All tickers combined (useful for yfinance downloads)
ALL_TICKERS = list(SECTOR_ETFS.keys()) + list(MACRO_PROXIES.keys())


# ============================================================================
# DATA FREQUENCY
# ============================================================================
FREQUENCY = "ME"  # Month-End


# ============================================================================
# MODEL PARAMETERS
# ============================================================================
N_REGIMES = 3  # Risk-On, Risk-Off, Reflation
RANDOM_STATE = 42  # For reproducibility

# Walk-forward cross-validation settings
MIN_TRAIN_YEARS = 5  # Minimum 5 years initial training
REFIT_FREQUENCY_YEARS = 1  # Refit model annually
WINDOW_TYPE = "expanding"  # Preserves rare regime examples


# ============================================================================
# STRATEGY PARAMETERS
# ============================================================================
REBALANCE_FREQUENCY = "ME"  # Monthly rebalancing
TOP_N_SECTORS = 3  # Hold top 3 sectors (concentrated active share)
TRANSACTION_COST_BPS = 5  # 5 basis points one-way, per trade

# Momentum benchmark parameters (Jegadeesh & Titman, 1993)
MOMENTUM_LOOKBACK_MONTHS = 6  # Lookback period
MOMENTUM_SKIP_MONTHS = 1  # Skip most recent month to avoid reversal


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
# These features will be engineered from MACRO_PROXIES
# (Listed here for documentation purposes)
FEATURE_NAMES = [
    "yield_curve_slope",      # ^TNX - ^FVX
    "credit_spread_proxy",    # HYG_price / LQD_price
    "breakeven_proxy",        # TIP_price / IEF_price
    "vix_level",              # ^VIX (log transformed)
    "vix_change",             # ^VIX 1-month change
    "short_rate",             # ^IRX
    "gold_momentum_3m",       # GLD 3-month return
    "oil_momentum_3m",        # USO 3-month return
    "credit_momentum_3m",     # HYG 3-month return (risk appetite)
]

# Number of features (used for model validation)
N_FEATURES = len(FEATURE_NAMES)


# ============================================================================
# OUTPUT FILES
# ============================================================================
# Names for saved files (optional - used if you save intermediate data)
SECTOR_PRICES_FILE = DATA_DIR / "sector_prices.parquet"
MACRO_PRICES_FILE = DATA_DIR / "macro_prices.parquet"
FEATURES_FILE = DATA_DIR / "features.parquet"
SECTOR_RETURNS_FILE = DATA_DIR / "sector_returns.parquet"
REGIME_LABELS_FILE = DATA_DIR / "regime_labels.parquet"
AVAILABLE_MASK_FILE = DATA_DIR / "available_mask.parquet"  # <-- ADD THIS LINE


# ============================================================================
# DEPENDENCY CHECK
# ============================================================================
def check_environment():
    """Quick check to verify imports work."""
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        from sklearn.mixture import GaussianMixture
        from hmmlearn import hmm
        print("All imports successful!")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    # Quick test when run directly
    print(f"Project configured with {len(SECTOR_ETFS)} sectors and {len(MACRO_PROXIES)} macro proxies.")
    print(f"Total tickers: {len(ALL_TICKERS)}")
    print(f"Data range: {START_DATE} to {END_DATE}")
    print(f"Features: {N_FEATURES}")
    
    # Check environment
    check_environment()