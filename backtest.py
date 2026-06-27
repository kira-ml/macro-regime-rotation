# backtest.py
"""
Strategy backtesting for macro-regime-rotation.
Implements regime rotation strategy and benchmarks with performance metrics.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from config import (
    TOP_N_SECTORS, TRANSACTION_COST_BPS, MOMENTUM_LOOKBACK_MONTHS,
    MOMENTUM_SKIP_MONTHS, OUTPUT_DIR
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_returns(prices):
    """Calculate monthly returns from price data."""
    returns = prices.pct_change().dropna()
    return returns


def select_top_sectors(regime_performance, regime, n_sectors=TOP_N_SECTORS):
    """
    Select top N sectors for a given regime based on historical performance.
    
    Parameters:
    -----------
    regime_performance : pd.DataFrame
        Average returns by regime and sector (from models.py)
    regime : int
        Current regime
    n_sectors : int
        Number of sectors to select
    
    Returns:
    --------
    list : Selected sector tickers
    """
    if regime not in regime_performance.index:
        # Fallback: equal weight all sectors
        return list(regime_performance.columns)
    
    # Get top N sectors for this regime
    sector_returns = regime_performance.loc[regime]
    top_sectors = sector_returns.nlargest(n_sectors).index.tolist()
    
    return top_sectors


def apply_transaction_costs(portfolio_returns, turnover, cost_bps=TRANSACTION_COST_BPS):
    """
    Apply transaction costs to portfolio returns.
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Gross portfolio returns
    turnover : pd.Series
        Portfolio turnover (fraction of portfolio traded)
    cost_bps : int
        Transaction cost in basis points (one-way)
    
    Returns:
    --------
    pd.Series : Net returns after transaction costs
    """
    cost = cost_bps / 10000  # Convert bps to decimal
    cost_impact = turnover * cost
    net_returns = portfolio_returns - cost_impact
    
    return net_returns


# ============================================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================================

def run_regime_strategy(sector_returns, predictions, regime_performance, 
                        available_mask=None, cost_bps=TRANSACTION_COST_BPS,
                        top_n=TOP_N_SECTORS):
    """
    Run the regime rotation strategy.
    
    Parameters:
    -----------
    sector_returns : pd.DataFrame
        Monthly sector returns
    predictions : pd.Series
        Regime predictions (index = dates)
    regime_performance : pd.DataFrame
        Historical sector performance by regime
    available_mask : pd.DataFrame, optional
        Boolean mask of available sectors
    cost_bps : int
        Transaction cost in basis points
    top_n : int
        Number of top sectors to hold
    
    Returns:
    --------
    dict : {
        'returns': pd.Series,  # Strategy returns
        'turnover': pd.Series,  # Portfolio turnover
        'weights': pd.DataFrame,  # Portfolio weights over time
        'sectors_held': dict  # Sectors held at each date
    }
    """
    # Align data
    common_idx = sector_returns.index.intersection(predictions.index)
    sector_returns = sector_returns.loc[common_idx]
    predictions = predictions.loc[common_idx]
    
    if available_mask is not None:
        available_mask = available_mask.loc[common_idx]
    
    # Initialize storage
    dates = common_idx
    weights = pd.DataFrame(0, index=dates, columns=sector_returns.columns)
    turnover = pd.Series(0.0, index=dates)
    sectors_held = {}
    
    prev_weights = pd.Series(0, index=sector_returns.columns)
    
    print(f"\nRunning regime strategy ({len(dates)} months)...")
    
    for i, date in enumerate(tqdm(dates, desc="Backtesting")):
        regime = predictions.loc[date]
        
        # Get available sectors at this date
        if available_mask is not None:
            available = available_mask.loc[date]
            available_sectors = available[available].index.tolist()
        else:
            available_sectors = sector_returns.columns.tolist()
        
        # Get top sectors for this regime (filter by available)
        if regime in regime_performance.index:
            regime_ranking = regime_performance.loc[regime].sort_values(ascending=False)
            # Filter to available sectors
            regime_ranking = regime_ranking[regime_ranking.index.isin(available_sectors)]
            selected = regime_ranking.head(top_n).index.tolist()
        else:
            # Fallback: equal weight available sectors
            selected = available_sectors[:min(top_n, len(available_sectors))]
        
        # Ensure we have at least one sector
        if not selected:
            selected = available_sectors[:1]
        
        # Equal weight selected sectors
        current_weights = pd.Series(0, index=sector_returns.columns)
        current_weights[selected] = 1.0 / len(selected)
        
        # Store
        weights.loc[date] = current_weights
        sectors_held[date] = selected
        
        # Calculate turnover (fraction of portfolio changed)
        if i > 0:
            turnover.loc[date] = (current_weights - prev_weights).abs().sum() / 2
        
        prev_weights = current_weights.copy()
    
    # Calculate portfolio returns
    # For each month, return = sum(weights_at_start * returns)
    portfolio_returns = pd.Series(0.0, index=dates)
    
    for i in range(len(dates)):
        if i == 0:
            continue
        date = dates[i]
        prev_date = dates[i-1]
        
        # Use weights from previous month to calculate returns
        w = weights.loc[prev_date]
        r = sector_returns.loc[date]
        portfolio_returns.loc[date] = (w * r).sum()
    
    # Apply transaction costs
    net_returns = apply_transaction_costs(portfolio_returns, turnover, cost_bps)
    
    return {
        'returns': net_returns,
        'turnover': turnover,
        'weights': weights,
        'sectors_held': sectors_held
    }


def run_equal_weight_benchmark(sector_returns, available_mask=None):
    """
    Equal weight benchmark: invest equally in all available sectors.
    """
    dates = sector_returns.index
    weights = pd.DataFrame(0, index=dates, columns=sector_returns.columns)
    portfolio_returns = pd.Series(0.0, index=dates)
    
    for i, date in enumerate(dates):
        if available_mask is not None:
            available = available_mask.loc[date]
            available_sectors = available[available].index.tolist()
        else:
            available_sectors = sector_returns.columns.tolist()
        
        if available_sectors:
            w = pd.Series(0, index=sector_returns.columns)
            w[available_sectors] = 1.0 / len(available_sectors)
            weights.loc[date] = w
    
    # Calculate returns (using weights at start of month)
    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]
        w = weights.loc[prev_date]
        r = sector_returns.loc[date]
        portfolio_returns.loc[date] = (w * r).sum()
    
    return {
        'returns': portfolio_returns,
        'weights': weights
    }


def run_momentum_benchmark(sector_returns, lookback=MOMENTUM_LOOKBACK_MONTHS, 
                           skip=MOMENTUM_SKIP_MONTHS, top_n=TOP_N_SECTORS,
                           available_mask=None):
    """
    Naive momentum strategy: buy top N sectors based on past performance.
    Implements Jegadeesh & Titman (1993) specification.
    """
    dates = sector_returns.index
    weights = pd.DataFrame(0, index=dates, columns=sector_returns.columns)
    portfolio_returns = pd.Series(0.0, index=dates)
    
    for i in range(lookback + skip, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]
        
        # Get available sectors
        if available_mask is not None:
            available = available_mask.loc[date]
            available_sectors = available[available].index.tolist()
        else:
            available_sectors = sector_returns.columns.tolist()
        
        # Calculate momentum score (returns from t-lookback-skip to t-skip)
        start_idx = i - lookback - skip
        end_idx = i - skip
        momentum = (1 + sector_returns.iloc[start_idx:end_idx]).prod() - 1
        momentum = momentum[momentum.index.isin(available_sectors)]
        
        # Select top N
        selected = momentum.nlargest(top_n).index.tolist()
        
        if not selected:
            selected = available_sectors[:min(top_n, len(available_sectors))]
        
        # Equal weight
        w = pd.Series(0, index=sector_returns.columns)
        w[selected] = 1.0 / len(selected)
        weights.loc[date] = w
        
        # Calculate return for this month
        r = sector_returns.loc[date]
        portfolio_returns.loc[date] = (w * r).sum()
    
    # First few months with no weights (use equal weight)
    for i in range(min(lookback + skip, len(dates))):
        date = dates[i]
        if available_mask is not None:
            available = available_mask.loc[date]
            available_sectors = available[available].index.tolist()
        else:
            available_sectors = sector_returns.columns.tolist()
        
        if available_sectors:
            w = pd.Series(0, index=sector_returns.columns)
            w[available_sectors] = 1.0 / len(available_sectors)
            weights.loc[date] = w
            
            if i > 0:
                prev_date = dates[i-1]
                prev_w = weights.loc[prev_date]
                r = sector_returns.loc[date]
                portfolio_returns.loc[date] = (prev_w * r).sum()
    
    return {
        'returns': portfolio_returns,
        'weights': weights
    }




def run_spy_benchmark(sector_returns, spy_prices):
    """
    Buy & Hold SPY benchmark.
    
    Parameters:
    -----------
    sector_returns : pd.DataFrame
        Monthly sector returns (used for index alignment)
    spy_prices : pd.Series or pd.DataFrame
        SPY monthly prices
    
    Returns:
    --------
    dict : {'returns': pd.Series, 'weights': pd.DataFrame}
    """
    # Convert to DataFrame if it's a Series
    if isinstance(spy_prices, pd.Series):
        spy_prices = spy_prices.to_frame('SPY')
    
    # Calculate SPY returns
    spy_returns = spy_prices.pct_change().dropna()
    
    # Align with sector_returns index
    common_idx = sector_returns.index.intersection(spy_returns.index)
    spy_returns = spy_returns.loc[common_idx]
    
    # Create dummy weights (SPY is 100% of the portfolio)
    weights = pd.DataFrame(0, index=common_idx, columns=sector_returns.columns)
    
    return {
        'returns': spy_returns['SPY'],
        'weights': weights
    }




# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calculate_metrics(returns, risk_free_rate=0.02):
    """
    Calculate comprehensive performance metrics.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    risk_free_rate : float
        Annual risk-free rate (default 2%)
    
    Returns:
    --------
    dict : Performance metrics
    """
    if returns.empty or returns.isna().all():
        return {
            'annual_return': np.nan,
            'volatility': np.nan,
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan,
            'calmar_ratio': np.nan,
            'win_rate': np.nan,
            'avg_monthly_return': np.nan,
            'total_return': np.nan,
            'n_months': 0
        }
    
    # Clean returns
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return {
            'annual_return': np.nan,
            'volatility': np.nan,
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan,
            'calmar_ratio': np.nan,
            'win_rate': np.nan,
            'avg_monthly_return': np.nan,
            'total_return': np.nan,
            'n_months': 0
        }
    
    n_months = len(returns_clean)
    
    # Total return
    total_return = (1 + returns_clean).prod() - 1
    
    # Annualized return (assuming 12 months per year)
    annual_return = (1 + total_return) ** (12 / n_months) - 1
    
    # Volatility (annualized)
    volatility = returns_clean.std() * np.sqrt(12)
    
    # Sharpe ratio
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else np.nan
    
    # Max drawdown
    cumulative = (1 + returns_clean).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
    
    # Win rate
    win_rate = (returns_clean > 0).mean()
    
    # Average monthly return
    avg_monthly_return = returns_clean.mean()
    
    return {
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'avg_monthly_return': avg_monthly_return,
        'total_return': total_return,
        'n_months': n_months
    }


def calculate_cumulative_returns(results_dict):
    """
    Calculate cumulative returns for multiple strategies.
    
    Parameters:
    -----------
    results_dict : dict
        {'strategy_name': {'returns': pd.Series}, ...}
    
    Returns:
    --------
    pd.DataFrame : Cumulative returns for all strategies
    """
    cumulative = pd.DataFrame()
    
    for name, result in results_dict.items():
        returns = result.get('returns', pd.Series())
        if not returns.empty:
            cumulative[name] = (1 + returns).cumprod()
    
    return cumulative


def compare_strategies(strategy_results):
    """
    Compare multiple strategies and create summary table.
    
    Parameters:
    -----------
    strategy_results : dict
        {'strategy_name': {'returns': pd.Series, ...}, ...}
    
    Returns:
    --------
    pd.DataFrame : Performance comparison table
    """
    metrics_list = []
    
    for name, result in strategy_results.items():
        returns = result.get('returns', pd.Series())
        metrics = calculate_metrics(returns)
        metrics['Strategy'] = name
        metrics_list.append(metrics)
    
    df = pd.DataFrame(metrics_list)
    
    # Reorder columns
    columns = ['Strategy', 'annual_return', 'volatility', 'sharpe_ratio', 
               'max_drawdown', 'calmar_ratio', 'win_rate', 'avg_monthly_return', 
               'total_return', 'n_months']
    df = df[columns]
    
    # Format for display
    df_formatted = df.copy()
    for col in ['annual_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 
                'calmar_ratio', 'win_rate', 'avg_monthly_return', 'total_return']:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A")
    
    return df, df_formatted


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_cumulative_returns(cumulative_returns, title="Cumulative Returns Comparison"):
    """
    Create interactive plot of cumulative returns.
    """
    fig = go.Figure()
    
    for col in cumulative_returns.columns:
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns[col],
            mode='lines',
            name=col,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".0%",
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98),
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_regime_timeline(predictions, sector_returns=None, cumulative_returns=None):
    """
    Create dual-panel chart: cumulative returns (top) and regime timeline (bottom).
    Uses 3-month rolling average to smooth the regime transitions.
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("Cumulative Returns", "Regime Timeline (3-Month Smoothed)")
    )
    
    # Top: Cumulative returns
    if cumulative_returns is not None:
        for col in cumulative_returns.columns:
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns[col],
                    mode='lines',
                    name=col,
                    line=dict(width=2)
                ),
                row=1, col=1
            )
    
    # Bottom: Regime timeline (smoothed with 3-month rolling average)
    dates = predictions.index
    regimes = predictions.values
    
    # Create a DataFrame with regime indicators
    regime_df = pd.DataFrame(index=dates)
    regime_labels = {
        0: 'Risk-On',
        1: 'Risk-Off',
        2: 'Reflation'
    }
    regime_colors = {
        0: '#1f77b4',  # Blue
        1: '#ff7f0e',  # Orange
        2: '#2ca02c',  # Green
    }
    
    # Create dummy probabilities (1 for predicted state)
    for state in range(3):
        regime_df[f'State_{state}'] = (regimes == state).astype(float)
    
    # Apply 3-month rolling average for smoothing
    regime_df_smoothed = regime_df.rolling(window=3, min_periods=1).mean()
    
    # Plot smoothed probabilities as stacked area
    for state in range(3):
        label = regime_labels.get(state, f'State {state}')
        color = regime_colors.get(state, '#808080')
        fig.add_trace(
            go.Scatter(
                x=regime_df_smoothed.index,
                y=regime_df_smoothed[f'State_{state}'],
                mode='lines',
                name=label,
                stackgroup='one',
                line=dict(width=0.5, color=color),
                fillcolor=color,
                opacity=0.8,
                hovertemplate='%{x|%b %Y}<br>%{y:.0%}<extra>%{fullData.name}</extra>'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title="Regime Rotation Strategy",
        height=800,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98)
    )
    
    fig.update_yaxes(title_text="Cumulative Return", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Regime Probability", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    return fig


def plot_drawdowns(drawdown_data, title="Drawdown Comparison"):
    """
    Plot drawdowns for multiple strategies.
    """
    fig = go.Figure()
    
    for col in drawdown_data.columns:
        fig.add_trace(go.Scatter(
            x=drawdown_data.index,
            y=drawdown_data[col],
            mode='lines',
            name=col,
            line=dict(width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98),
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_sector_heatmap(weights, title="Sector Allocations Over Time"):
    """
    Create heatmap of sector allocations.
    """
    # Transpose for heatmap
    weights_t = weights.T
    
    fig = go.Figure(data=go.Heatmap(
        z=weights_t.values,
        x=weights_t.columns,
        y=weights_t.index,
        colorscale='RdYlGn',
        zmid=0.3,
        hoverongaps=False,
        colorbar=dict(title="Weight")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sector",
        height=500,
        template='plotly_white'
    )
    
    return fig


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def run_full_backtest(features, sector_returns, available_mask, 
                      hmm_predictions, gmm_predictions, 
                      hmm_regime_performance, gmm_regime_performance,
                      save_plots=True):
    """
    Run complete backtest with all strategies and generate reports.
    """
    from config import BENCHMARK_ETF, DATA_DIR
    
    print("\n" + "="*60)
    print("BACKTEST: REGIME ROTATION STRATEGY")
    print("="*60)
    
    # Load SPY data from the saved macro/benchmark data
    import pandas as pd
    macro_prices = pd.read_parquet(DATA_DIR / "macro_prices.parquet")
    spy_prices = macro_prices[BENCHMARK_ETF] if BENCHMARK_ETF in macro_prices.columns else None
    
    # Run all strategies
    results = {}
    
    # 1. HMM Regime Strategy
    print("\n[1/5] Running HMM Regime Strategy...")
    hmm_strategy = run_regime_strategy(
        sector_returns, hmm_predictions, hmm_regime_performance, available_mask
    )
    results['HMM Regime'] = hmm_strategy
    
    # 2. GMM Regime Strategy
    print("\n[2/5] Running GMM Regime Strategy...")
    gmm_strategy = run_regime_strategy(
        sector_returns, gmm_predictions, gmm_regime_performance, available_mask
    )
    results['GMM Regime'] = gmm_strategy
    
    # 3. Equal Weight Benchmark
    print("\n[3/5] Running Equal Weight Benchmark...")
    ew_strategy = run_equal_weight_benchmark(sector_returns, available_mask)
    results['Equal Weight'] = ew_strategy
    
    # 4. Momentum Benchmark
    print("\n[4/5] Running Momentum Benchmark...")
    mom_strategy = run_momentum_benchmark(sector_returns, available_mask=available_mask)
    results['Momentum'] = mom_strategy
    
    # 5. SPY Buy & Hold Benchmark
    print("\n[5/5] Running SPY Buy & Hold Benchmark...")
    if spy_prices is not None and not spy_prices.empty:
        bh_strategy = run_spy_benchmark(sector_returns, spy_prices)
        results['SPY (Buy & Hold)'] = bh_strategy
    else:
        print("  WARNING: SPY data not found. Falling back to equal-weight benchmark.")
        bh_strategy = run_equal_weight_benchmark(sector_returns, available_mask)
        results['Buy & Hold (EW)'] = bh_strategy
    
    # Calculate performance metrics
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    metrics_df, metrics_formatted = compare_strategies(results)
    print("\n", metrics_formatted.to_string(index=False))
    
    # Calculate cumulative returns
    cumulative = calculate_cumulative_returns(results)
    
    # Calculate drawdowns
    drawdowns = pd.DataFrame()
    for name, result in results.items():
        returns = result.get('returns', pd.Series())
        if not returns.empty:
            cum = (1 + returns).cumprod()
            running_max = cum.expanding().max()
            drawdown = (cum - running_max) / running_max
            drawdowns[name] = drawdown
    
    # Save outputs
    if save_plots:
        print("\nGenerating plots...")
        
        # 1. Cumulative returns
        fig1 = plot_cumulative_returns(cumulative)
        fig1.write_html(OUTPUT_DIR / "cumulative_returns.html")
        fig1.write_image(OUTPUT_DIR / "cumulative_returns.png")
        print(f"  Saved: {OUTPUT_DIR}/cumulative_returns.html")
        
        # 2. Regime timeline
        fig2 = plot_regime_timeline(hmm_predictions, sector_returns, cumulative)
        fig2.write_html(OUTPUT_DIR / "regime_timeline.html")
        fig2.write_image(OUTPUT_DIR / "regime_timeline.png")
        print(f"  Saved: {OUTPUT_DIR}/regime_timeline.html")
        
        # 3. Drawdowns
        fig3 = plot_drawdowns(drawdowns)
        fig3.write_html(OUTPUT_DIR / "drawdowns.html")
        fig3.write_image(OUTPUT_DIR / "drawdowns.png")
        print(f"  Saved: {OUTPUT_DIR}/drawdowns.html")
        
        # 4. Sector heatmap (HMM strategy)
        fig4 = plot_sector_heatmap(hmm_strategy['weights'])
        fig4.write_html(OUTPUT_DIR / "sector_allocations.html")
        fig4.write_image(OUTPUT_DIR / "sector_allocations.png")
        print(f"  Saved: {OUTPUT_DIR}/sector_allocations.html")
    
    # Return everything
    return {
        'results': results,
        'metrics': metrics_df,
        'metrics_formatted': metrics_formatted,
        'cumulative': cumulative,
        'drawdowns': drawdowns,
        'hmm_strategy': hmm_strategy,
        'gmm_strategy': gmm_strategy
    }
# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing backtest module...")
    
    # Load data and models
    from data import get_regime_data
    from models import walk_forward_predictions, sector_performance_by_regime
    
    # Get data
    data = get_regime_data()
    features = data['features']
    sector_returns = data['sector_returns']
    available_mask = data['available_mask']
    
    # Run HMM
    hmm_results = walk_forward_predictions(
        features, model_type='hmm', min_train_years=5, refit_years=1
    )
    hmm_predictions = hmm_results['predictions']
    
    # Run GMM
    gmm_results = walk_forward_predictions(
        features, model_type='gmm', min_train_years=5, refit_years=1
    )
    gmm_predictions = gmm_results['predictions']
    
    # Get regime performance
    hmm_performance = sector_performance_by_regime(sector_returns, hmm_predictions)
    gmm_performance = sector_performance_by_regime(sector_returns, gmm_predictions)
    
    # Run backtest
    backtest_results = run_full_backtest(
        features, sector_returns, available_mask,
        hmm_predictions, gmm_predictions,
        hmm_performance, gmm_performance,
        save_plots=True
    )
    
    print("\n✅ backtest.py test complete!")