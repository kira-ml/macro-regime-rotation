# data.py
"""
Data acquisition and feature engineering for macro-regime-rotation.
All data sourced from yfinance. Handles:
- Fetching sector ETFs and macro proxies
- Resampling to month-end
- Dynamic sector universe (handles XLRE, XLC inception dates)
- Feature engineering with NO look-ahead bias
- Train/test split for walk-forward validation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from config import (
    START_DATE, END_DATE, SECTOR_ETFS, MACRO_PROXIES, ALL_TICKERS,
    FEATURE_NAMES, FREQUENCY, MIN_TRAIN_YEARS,
    SECTOR_PRICES_FILE, MACRO_PRICES_FILE, FEATURES_FILE, SECTOR_RETURNS_FILE,
    AVAILABLE_MASK_FILE, DATA_DIR, OUTPUT_DIR
)


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_data(tickers, start=START_DATE, end=END_DATE, progress=True):
    """
    Fetch adjusted close prices for a list of tickers from yfinance.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start : str
        Start date in 'YYYY-MM-DD' format
    end : str
        End date in 'YYYY-MM-DD' format
    progress : bool
        Show progress bar
    
    Returns:
    --------
    pd.DataFrame : Adjusted close prices, columns = tickers
    """
    print(f"Fetching {len(tickers)} tickers from {start} to {end}...")
    
    # Download with progress bar
    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=progress,
        auto_adjust=False  # Keep all columns, we'll extract Adj Close
    )
    
    # Extract Adjusted Close
    if isinstance(data.columns, pd.MultiIndex):
        # Multiple tickers - extract Adj Close
        adj_close = data['Adj Close']
    else:
        # Single ticker - handle edge case
        adj_close = pd.DataFrame(data['Adj Close'])
        adj_close.columns = tickers
    
    print(f"Fetched {len(adj_close)} days of data for {len(adj_close.columns)} tickers")
    return adj_close


def resample_to_monthly(df):
    """
    Resample daily data to month-end frequency.
    Uses last available price of each month.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Daily price data
    
    Returns:
    --------
    pd.DataFrame : Month-end prices
    """
    # Resample to month-end, taking the last available value
    monthly = df.resample(FREQUENCY).last()
    
    # Drop any rows that are all NaN
    monthly = monthly.dropna(how='all')
    
    print(f"Resampled to {len(monthly)} monthly observations")
    return monthly


# ============================================================================
# SECTOR UNIVERSE ALIGNMENT
# ============================================================================

def get_available_sectors(sector_prices, date):
    """
    Get list of sectors available at a given date.
    Handles ETFs that launched later (XLRE: Oct 2015, XLC: Jun 2018).
    
    Parameters:
    -----------
    sector_prices : pd.DataFrame
        Monthly sector prices with all sectors
    date : pd.Timestamp
        Date to check availability
    
    Returns:
    --------
    list : Available sector tickers at that date
    """
    # Get row at date
    row = sector_prices.loc[date]
    
    # Find columns that are not NaN
    available = row.dropna().index.tolist()
    
    return available


def create_dynamic_universe(sector_prices):
    """
    Create a DataFrame tracking which sectors are available at each date.
    
    Parameters:
    -----------
    sector_prices : pd.DataFrame
        Monthly sector prices
    
    Returns:
    --------
    pd.DataFrame : Boolean mask of available sectors
    """
    # Create boolean mask: True where price is not NaN
    available_mask = sector_prices.notna()
    
    # Add metadata about when each sector became available
    n_available = available_mask.sum(axis=1)
    
    print(f"Dynamic universe ranges from {n_available.min()} to {n_available.max()} sectors")
    print(f"Sector availability:")
    for col in sector_prices.columns:
        first_date = sector_prices[col].first_valid_index()
        print(f"  {col}: {first_date.strftime('%Y-%m-%d') if first_date else 'Never'}")
    
    return available_mask


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(macro_prices):
    """
    Engineer all features from macro proxy prices.
    Returns a DataFrame with clean, interpretable features.
    
    Parameters:
    -----------
    macro_prices : pd.DataFrame
        Monthly prices for all macro proxies
    
    Returns:
    --------
    pd.DataFrame : Engineered features (standardized)
    """
    print("Engineering features from macro proxies...")
    
    # Create a copy to avoid modifying original
    df = macro_prices.copy()
    
    # Ensure we have all required columns
    required = ['^TNX', '^FVX', '^IRX', '^VIX', 'LQD', 'HYG', 'TIP', 'IEF', 'GLD', 'USO']
    missing = [r for r in required if r not in df.columns]
    if missing:
        print(f"WARNING: Missing macro proxies: {missing}")
    
    # Initialize features DataFrame
    features = pd.DataFrame(index=df.index)
    
    # 1. Yield Curve Slope: 10Y - 5Y
    if '^TNX' in df and '^FVX' in df:
        features['yield_curve_slope'] = df['^TNX'] - df['^FVX']
        print("  ✓ yield_curve_slope")
    
    # 2. Credit Spread Proxy: HYG / LQD (risk appetite)
    if 'HYG' in df and 'LQD' in df:
        features['credit_spread_proxy'] = df['HYG'] / df['LQD']
        print("  ✓ credit_spread_proxy")
    
    # 3. Breakeven Inflation Proxy: TIP / IEF
    if 'TIP' in df and 'IEF' in df:
        features['breakeven_proxy'] = df['TIP'] / df['IEF']
        print("  ✓ breakeven_proxy")
    
    # 4. VIX Level (log transform to handle non-normality)
    if '^VIX' in df:
        # Add small constant to avoid log(0)
        features['vix_level'] = np.log(df['^VIX'] + 1)
        print("  ✓ vix_level (log)")
    
    # 5. VIX 1-Month Change
    if '^VIX' in df:
        features['vix_change'] = df['^VIX'].pct_change()
        print("  ✓ vix_change")
    
    # 6. Short Rate (T-Bill)
    if '^IRX' in df:
        features['short_rate'] = df['^IRX']
        print("  ✓ short_rate")
    
    # 7. Gold 3-Month Momentum
    if 'GLD' in df:
        features['gold_momentum_3m'] = df['GLD'].pct_change(periods=3)
        print("  ✓ gold_momentum_3m")
    
    # 8. Oil 3-Month Momentum
    if 'USO' in df:
        features['oil_momentum_3m'] = df['USO'].pct_change(periods=3)
        print("  ✓ oil_momentum_3m")
    
    # 9. Credit 3-Month Momentum (risk appetite trend)
    if 'HYG' in df:
        features['credit_momentum_3m'] = df['HYG'].pct_change(periods=3)
        print("  ✓ credit_momentum_3m")
    
    # Drop rows with NaN (first few months where we don't have enough history)
    features = features.dropna()
    
    print(f"Features engineered: {len(features)} months, {len(features.columns)} features")
    
    return features


def standardize_features(features, fit_params=None):
    """
    Standardize features using rolling statistics to avoid look-ahead bias.
    
    Parameters:
    -----------
    features : pd.DataFrame
        Raw features
    fit_params : tuple, optional
        (mean, std) to use for transformation (for test set)
    
    Returns:
    --------
    pd.DataFrame : Standardized features
    dict : Fitted parameters (mean, std) for later use
    """
    if fit_params is not None:
        # Use provided parameters (for test set)
        mean, std = fit_params
        standardized = (features - mean) / std
        return standardized, fit_params
    
    # Fit on training data
    mean = features.mean()
    std = features.std()
    
    # Avoid division by zero
    std = std.replace(0, 1)
    
    standardized = (features - mean) / std
    
    return standardized, {'mean': mean, 'std': std}


# ============================================================================
# TRAIN/TEST SPLIT (Walk-Forward Aware)
# ============================================================================

def get_walk_forward_splits(features, min_train_years=MIN_TRAIN_YEARS, refit_years=1):
    """
    Generate train/test split dates for walk-forward validation.
    
    Parameters:
    -----------
    features : pd.DataFrame
        Monthly features with datetime index
    min_train_years : int
        Minimum years for initial training
    refit_years : int
        Refit frequency in years
    
    Returns:
    --------
    list : List of (train_start, train_end, test_start, test_end) tuples
    """
    dates = features.index
    n_months = len(dates)
    
    # Convert years to months
    min_train_months = min_train_years * 12
    refit_months = refit_years * 12
    
    splits = []
    
    # Start with minimum training period
    train_start_idx = 0
    train_end_idx = min_train_months
    
    while train_end_idx + 1 < n_months:
        # Test is next month after training
        test_start_idx = train_end_idx
        test_end_idx = min(train_end_idx + refit_months, n_months)
        
        splits.append((
            dates[train_start_idx],  # train_start
            dates[train_end_idx],    # train_end
            dates[test_start_idx],   # test_start (regime prediction date)
            dates[test_end_idx - 1]  # test_end (last prediction date)
        ))
        
        # Move window forward
        train_end_idx = min(train_end_idx + refit_months, n_months - 1)
    
    print(f"Generated {len(splits)} walk-forward splits")
    return splits


# ============================================================================
# SAVE/LOAD HELPERS
# ============================================================================

def save_dataframe(df, filepath, format='parquet'):
    """
    Save DataFrame with fallback to CSV if parquet not available.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filepath : Path
        Path to save to
    format : str
        'parquet' or 'csv'
    """
    try:
        if format == 'parquet':
            df.to_parquet(filepath)
            print(f"  Saved to {filepath} (parquet)")
        else:
            csv_path = filepath.with_suffix('.csv')
            df.to_csv(csv_path)
            print(f"  Saved to {csv_path} (csv)")
    except (ImportError, Exception) as e:
        # Fallback to CSV
        csv_path = filepath.with_suffix('.csv')
        df.to_csv(csv_path)
        print(f"  WARNING: {e}")
        print(f"  Saved to {csv_path} (csv fallback)")


def load_dataframe(filepath):
    """
    Load DataFrame with fallback from CSV if parquet not available.
    
    Parameters:
    -----------
    filepath : Path
        Path to load from
    
    Returns:
    --------
    pd.DataFrame : Loaded DataFrame
    """
    try:
        return pd.read_parquet(filepath)
    except (ImportError, FileNotFoundError):
        # Try CSV
        csv_path = filepath.with_suffix('.csv')
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def get_regime_data(force_refresh=False):
    """
    Main entry point for data acquisition and feature engineering.
    Fetches all data, processes it, and returns ready-to-use features and returns.
    
    Features are shifted by 1 month BEFORE any standardization to ensure
    zero look-ahead bias. Standardization is performed per-fold during
    walk-forward validation in models.py, not here.
    
    Parameters:
    -----------
    force_refresh : bool
        If True, re-download data even if saved files exist
    
    Returns:
    --------
    dict : {
        'features': pd.DataFrame,  # Raw engineered features (shifted, not standardized)
        'sector_returns': pd.DataFrame,  # Monthly sector returns
        'available_mask': pd.DataFrame,  # Boolean mask of available sectors
        'dates': pd.DatetimeIndex,  # All dates
        'metadata': dict  # Additional info
    }
    """
    print("=" * 60)
    print("MACRO-REGIME-ROTATION: DATA PIPELINE")
    print("=" * 60)
    
    # --- Auto-refresh if cached data is stale (> 30 days old) ---
    if not force_refresh and SECTOR_PRICES_FILE.exists():
        import datetime
        mod_time = datetime.datetime.fromtimestamp(SECTOR_PRICES_FILE.stat().st_mtime)
        if (datetime.datetime.now() - mod_time).days > 30:
            print("Cached data is stale (>30 days). Forcing refresh...")
            force_refresh = True
    
    # Check if we have saved data
    if not force_refresh and FEATURES_FILE.exists() and SECTOR_RETURNS_FILE.exists():
        print("Loading saved data...")
        try:
            features = load_dataframe(FEATURES_FILE)
            sector_returns = load_dataframe(SECTOR_RETURNS_FILE)
            available_mask = load_dataframe(AVAILABLE_MASK_FILE)
            
            print(f"Loaded {len(features)} months of data")
            
            return {
                'features': features,
                'sector_returns': sector_returns,
                'available_mask': available_mask,
                'dates': features.index,
                'metadata': {
                    'n_months': len(features),
                    'n_sectors': len(sector_returns.columns),
                    'n_features': len(features.columns),
                    'data_source': 'yfinance'
                }
            }
        except Exception as e:
            print(f"Could not load saved data: {e}")
            print("Re-downloading...")
    
    # ---------- STEP 1: Fetch all data ----------
    print("\n[1/5] Fetching data from yfinance...")
    all_prices = fetch_data(ALL_TICKERS, START_DATE, END_DATE)
    
    # Split into sectors and macro
    sector_prices = all_prices[SECTOR_ETFS.keys()]
    macro_prices = all_prices[MACRO_PROXIES.keys()]
    
    # Add SPY to macro_prices (so it gets saved for the backtest)
    from config import BENCHMARK_ETF
    if BENCHMARK_ETF in all_prices.columns:
        macro_prices[BENCHMARK_ETF] = all_prices[BENCHMARK_ETF]
        print(f"  Added {BENCHMARK_ETF} to macro dataset for benchmarking")
    
    print(f"  Sectors: {len(sector_prices.columns)}")
    print(f"  Macro + Benchmarks: {len(macro_prices.columns)}")
    
    # ---------- STEP 2: Resample to monthly ----------
    print("\n[2/5] Resampling to month-end...")
    sector_prices = resample_to_monthly(sector_prices)
    macro_prices = resample_to_monthly(macro_prices)
    
    # Align indices
    common_idx = sector_prices.index.intersection(macro_prices.index)
    sector_prices = sector_prices.loc[common_idx]
    macro_prices = macro_prices.loc[common_idx]
    
    # ---------- STEP 3: Create dynamic universe ----------
    print("\n[3/5] Creating dynamic sector universe...")
    available_mask = create_dynamic_universe(sector_prices)
    
    # ---------- STEP 4: Engineer features ----------
    print("\n[4/5] Engineering features...")
    features_raw = engineer_features(macro_prices)
    
    # Align with sector data
    common_idx = features_raw.index.intersection(sector_prices.index)
    features_raw = features_raw.loc[common_idx]
    sector_prices = sector_prices.loc[common_idx]
    available_mask = available_mask.loc[common_idx]
    
    # Drop NaN explicitly
    features_raw = features_raw.dropna()
    
    # ---------- STEP 5: Compute sector returns ----------
    print("\n[5/5] Computing sector returns...")
    sector_returns = sector_prices.pct_change().dropna()
    
    # Align raw features with sector returns (drop first month where returns are NaN)
    common_idx = features_raw.index.intersection(sector_returns.index)
    features_raw = features_raw.loc[common_idx]
    sector_returns = sector_returns.loc[common_idx]
    available_mask = available_mask.loc[common_idx]
    
    # ---------- CRITICAL: Shift features BEFORE any standardization ----------
    # Shift by 1 month so features at time t use only data available through t.
    # This must happen before standardization so mean/std don't leak future info.
    # Standardization is handled per-fold in models.py during walk-forward validation.
    features_raw = features_raw.shift(1).dropna()
    
    # Re-align after shift
    common_idx = features_raw.index.intersection(sector_returns.index)
    features = features_raw.loc[common_idx]
    sector_returns = sector_returns.loc[common_idx]
    available_mask = available_mask.loc[common_idx]
    
    # ---------- SAVE DATA ----------
    print("\nSaving processed data...")
    save_dataframe(sector_prices, SECTOR_PRICES_FILE)
    save_dataframe(macro_prices, MACRO_PRICES_FILE)
    save_dataframe(features, FEATURES_FILE)
    save_dataframe(sector_returns, SECTOR_RETURNS_FILE)
    save_dataframe(available_mask, AVAILABLE_MASK_FILE)
    
    print(f"  Saved to {DATA_DIR}")
    print(f"\n✅ Data pipeline complete!")
    print(f"  {len(features)} months, {len(features.columns)} features")
    print(f"  {len(sector_returns.columns)} sectors")
    print(f"  Date range: {features.index[0].strftime('%Y-%m-%d')} to {features.index[-1].strftime('%Y-%m-%d')}")
    
    return {
        'features': features,
        'sector_returns': sector_returns,
        'available_mask': available_mask,
        'dates': features.index,
        'metadata': {
            'n_months': len(features),
            'n_sectors': len(sector_returns.columns),
            'n_features': len(features.columns),
            'data_source': 'yfinance',
            'sector_etfs': list(SECTOR_ETFS.keys()),
            'macro_proxies': list(MACRO_PROXIES.keys()),
            'feature_names': list(features.columns)
        }
    }

# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing data pipeline...")
    data = get_regime_data()
    print("\nData summary:")
    print(f"  Features shape: {data['features'].shape}")
    print(f"  Returns shape: {data['sector_returns'].shape}")
    print(f"  Available mask shape: {data['available_mask'].shape}")
    print("\nFirst 5 rows of features:")
    print(data['features'].head())
    print("\n✅ data.py test complete!")