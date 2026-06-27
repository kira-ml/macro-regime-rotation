"""
Evaluation module for macro-regime-rotation strategy.
Implements core performance metrics, regime-conditional analysis,
and quant-finance-grade Matplotlib/Seaborn visualizations for LinkedIn.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

from config import OUTPUT_DIR

# ============================================================================
# QUANT FINANCE STYLE CONFIGURATION
# ============================================================================
def set_style():
    """Set global matplotlib style for institutional quant publications."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 150,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'grid.color': '#d9d9d9',
        'grid.linestyle': '-',
        'grid.alpha': 0.6,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
    })

# Call this once when module loads
set_style()

# Define Institutional Quant Color Palette (High Contrast)
QUANT_COLORS = {
    'hmm': '#1f77b4',      # Deep Blue (Primary strategy)
    'gmm': '#ff7f0e',      # Orange (Baseline)
    'spy': '#2ca02c',      # Green (Market benchmark)
    'momentum': '#d62728', # Red (Momentum)
    'equal': '#9467bd',    # Purple (Equal weight)
    'risk_on': '#2ecc71',  # Emerald Green
    'risk_off': '#e74c3c', # Crimson Red
    'reflation': '#3498db' # Sky Blue
}

# ============================================================================
# TIER 1: CORE METRICS (Non-Negotiable)
# ============================================================================

def compute_metrics(returns_series, risk_free_rate=0.03, annualization=12):
    """Compute standard performance metrics for strategy evaluation."""
    returns = returns_series.dropna()
    n_periods = len(returns)
    
    if n_periods == 0:
        return {k: np.nan for k in [
            'annual_return', 'annual_volatility', 'sharpe_ratio',
            'max_drawdown', 'calmar_ratio', 'win_rate',
            'cumulative_return', 'n_months'
        ]}
    
    # 1. Annualized Return
    cumulative_return = (1 + returns).prod()
    annual_return = cumulative_return ** (annualization / n_periods) - 1
    
    # 2. Annualized Volatility
    annual_vol = returns.std() * np.sqrt(annualization)
    
    # 3. Sharpe Ratio (The most important metric)
    monthly_rf = (1 + risk_free_rate) ** (1 / annualization) - 1
    excess_returns = returns - monthly_rf
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(annualization) if excess_returns.std() > 0 else np.nan
    
    # 4. Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    # 5. Calmar Ratio
    calmar = annual_return / abs(max_dd) if max_dd != 0 else np.nan
    
    # 6. Win Rate
    win_rate = (returns > 0).mean()
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'cumulative_return': cumulative_return - 1,
        'n_months': n_periods,
    }


# ============================================================================
# TIER 2: DIFFERENTIATING METRICS
# ============================================================================

def compute_regime_conditional_sharpe(returns, regime_labels, risk_free_rate=0.03):
    """Compute Sharpe ratio conditional on each regime."""
    monthly_rf = (1 + risk_free_rate) ** (1 / 12) - 1
    excess_returns = returns - monthly_rf
    
    results = []
    for regime in sorted(regime_labels.unique()):
        mask = regime_labels == regime
        regime_excess = excess_returns[mask]
        
        if len(regime_excess) >= 12:
            regime_sharpe = (regime_excess.mean() / regime_excess.std()) * np.sqrt(12) if regime_excess.std() > 0 else np.nan
        else:
            regime_sharpe = np.nan
        
        results.append({
            'regime': regime,
            'sharpe': regime_sharpe,
            'n_months': len(regime_excess),
            'pct_of_sample': len(regime_excess) / len(excess_returns) * 100,
            'avg_excess_return': regime_excess.mean(),
            'volatility': regime_excess.std() * np.sqrt(12) if len(regime_excess) > 0 else np.nan
        })
    
    return pd.DataFrame(results)


def compute_rolling_sharpe(returns, window_years=3, risk_free_rate=0.03):
    """Compute rolling Sharpe ratio over a rolling window."""
    window_months = window_years * 12
    monthly_rf = (1 + risk_free_rate) ** (1 / 12) - 1
    excess_returns = returns - monthly_rf
    
    rolling_sharpe = pd.Series(index=returns.index, dtype=float)
    
    for i in range(window_months, len(returns) + 1):
        window = excess_returns.iloc[i - window_months:i]
        if len(window) >= 12 and window.std() > 0:
            sharpe = (window.mean() / window.std()) * np.sqrt(12)
            rolling_sharpe.iloc[i - 1] = sharpe
    
    return rolling_sharpe


def compute_turnover(weights_history):
    """Compute average monthly turnover from a history of portfolio weights."""
    if weights_history.empty or len(weights_history) < 2:
        return np.nan
    
    weight_changes = weights_history.diff().abs().sum(axis=1)
    avg_turnover = weight_changes.mean()
    
    return avg_turnover


# ============================================================================
# TIER 3: COMPARISON TABLE
# ============================================================================

def create_comparison_table(strategy_results, risk_free_rate=0.03):
    """Create the single most important table: strategy comparison."""
    rows = []
    
    for name, result in strategy_results.items():
        if isinstance(result, dict):
            returns = result.get('returns', pd.Series())
        else:
            returns = result
        
        if returns is None or returns.empty:
            continue
            
        metrics = compute_metrics(returns, risk_free_rate)
        metrics['Strategy'] = name
        rows.append(metrics)
    
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    
    df = pd.DataFrame(rows)
    columns = ['Strategy', 'annual_return', 'annual_volatility', 'sharpe_ratio',
               'max_drawdown', 'calmar_ratio', 'win_rate', 'cumulative_return', 'n_months']
    df = df[[c for c in columns if c in df.columns]]
    
    df_display = df.copy()
    for col in ['annual_return', 'annual_volatility', 'max_drawdown', 
                'calmar_ratio', 'win_rate', 'cumulative_return']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A")
    for col in ['sharpe_ratio']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
    
    return df, df_display


# ============================================================================
# VISUALIZATIONS (Quant Finance Style)
# ============================================================================

def plot_hero_chart(cumulative_returns, predictions, regime_labels=None,
                    annotations=None, title="Regime Rotation Strategy"):
    """
    Plot 1: The Hero Chart — Cumulative returns with regime background.
    Institutional quant style.
    """
    if regime_labels is None:
        regime_labels = {0: 'Risk-On', 1: 'Risk-Off', 2: 'Reflation'}
    
    # Set up color palette
    line_colors = [QUANT_COLORS['hmm'], QUANT_COLORS['gmm'], QUANT_COLORS['spy'], 
                   QUANT_COLORS['momentum'], QUANT_COLORS['equal']]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                    gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(hspace=0.15)
    
    # --- Top Panel: Cumulative Returns ---
    for i, col in enumerate(cumulative_returns.columns):
        # Assign specific colors based on strategy name for consistency
        if 'HMM' in col:
            color = QUANT_COLORS['hmm']
        elif 'GMM' in col:
            color = QUANT_COLORS['gmm']
        elif 'SPY' in col:
            color = QUANT_COLORS['spy']
        elif 'Momentum' in col:
            color = QUANT_COLORS['momentum']
        else:
            color = line_colors[i % len(line_colors)]
            
        ax1.plot(cumulative_returns.index, cumulative_returns[col], 
                label=col, linewidth=2.5, color=color)
    
    # Add event annotations
    if annotations:
        y_max = cumulative_returns.max().max()
        for ann in annotations:
            date = pd.to_datetime(ann['date'])
            ax1.axvline(x=date, color='#333333', linestyle='--', alpha=0.6, linewidth=1.5)
            ax1.text(date, y_max * 0.93, ann['text'], rotation=0, 
                    color='#333333', fontsize=11, ha='center', weight='bold', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
    
    ax1.set_ylabel('Cumulative Return', fontsize=12, weight='bold')
    ax1.set_title(title, fontsize=18, weight='bold', pad=20)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax1.grid(True, alpha=0.4)
    
    # --- Bottom Panel: Regime Timeline ---
    dates = predictions.index
    regimes = predictions.values
    
    # Draw regime blocks
    current_regime = regimes[0]
    start_idx = 0
    
    for i in range(1, len(regimes)):
        if regimes[i] != current_regime:
            color = QUANT_COLORS.get(['risk_on', 'risk_off', 'reflation'][current_regime], 'gray')
            ax2.axvspan(dates[start_idx], dates[i-1], color=color, alpha=0.3, lw=0)
            current_regime = regimes[i]
            start_idx = i
    # Draw last block
    color = QUANT_COLORS.get(['risk_on', 'risk_off', 'reflation'][current_regime], 'gray')
    ax2.axvspan(dates[start_idx], dates[-1], color=color, alpha=0.3, lw=0)
    
    # Create custom legend for regimes
    regime_patches = [mpatches.Patch(color=QUANT_COLORS['risk_on'], alpha=0.3, label='Risk-On'),
                     mpatches.Patch(color=QUANT_COLORS['risk_off'], alpha=0.3, label='Risk-Off'),
                     mpatches.Patch(color=QUANT_COLORS['reflation'], alpha=0.3, label='Reflation')]
    
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_ylabel('Regime', fontsize=12, weight='bold')
    ax2.set_xlabel('Date', fontsize=12, weight='bold')
    
    # Add regime legend to the bottom
    ax2.legend(handles=regime_patches, loc='center left', frameon=False, 
              fontsize=10, ncol=3, bbox_to_anchor=(0.02, 0.5))
    
    # Format dates
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    return fig


def plot_regime_heatmap(regime_performance, title="Sector Performance by Regime"):
    """
    Plot 2: Regime Characterization Heatmap using Seaborn.
    """
    df = regime_performance.copy()
    # Normalize for better visualization (relative performance within each regime)
    df_normalized = df.div(df.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(df_normalized, annot=df.round(4), fmt='.2%', 
                cmap='RdYlGn', center=df_normalized.values.mean(),
                cbar_kws={'label': 'Relative Performance (%)'},
                ax=ax, annot_kws={'size': 11})
    
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Sector', fontsize=12, weight='bold')
    ax.set_ylabel('Regime', fontsize=12, weight='bold')
    ax.set_yticklabels([f"Regime {i}" for i in df.index], rotation=0, fontsize=11)
    ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_drawdown_comparison(drawdowns, title="Drawdown Comparison"):
    """
    Plot 3: Underwater Plot — Drawdown comparison.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    line_colors = [QUANT_COLORS['hmm'], QUANT_COLORS['gmm'], QUANT_COLORS['spy'], 
                   QUANT_COLORS['momentum'], QUANT_COLORS['equal']]
    
    for i, col in enumerate(drawdowns.columns):
        # Assign specific colors based on strategy name
        if 'HMM' in col:
            color = QUANT_COLORS['hmm']
        elif 'GMM' in col:
            color = QUANT_COLORS['gmm']
        elif 'SPY' in col:
            color = QUANT_COLORS['spy']
        elif 'Momentum' in col:
            color = QUANT_COLORS['momentum']
        else:
            color = line_colors[i % len(line_colors)]
            
        ax.fill_between(drawdowns.index, drawdowns[col], 0, 
                        color=color, alpha=0.15, label=col)
        ax.plot(drawdowns.index, drawdowns[col], color=color, 
               linewidth=2, alpha=0.9)
    
    ax.axhline(0, color='#333333', linestyle='-', linewidth=1)
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, weight='bold')
    ax.set_ylabel('Drawdown', fontsize=12, weight='bold')
    ax.legend(loc='lower left', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.4)
    
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    return fig


def plot_rolling_sharpe(rolling_sharpe_dict, title="Rolling 3-Year Sharpe Ratio"):
    """
    Plot 4: Rolling Sharpe Ratio.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    line_colors = [QUANT_COLORS['hmm'], QUANT_COLORS['gmm'], QUANT_COLORS['spy'], 
                   QUANT_COLORS['momentum'], QUANT_COLORS['equal']]
    
    for i, (name, sharpe_series) in enumerate(rolling_sharpe_dict.items()):
        # Assign specific colors based on strategy name
        if 'HMM' in name:
            color = QUANT_COLORS['hmm']
        elif 'GMM' in name:
            color = QUANT_COLORS['gmm']
        elif 'SPY' in name:
            color = QUANT_COLORS['spy']
        elif 'Momentum' in name:
            color = QUANT_COLORS['momentum']
        else:
            color = line_colors[i % len(line_colors)]
            
        if sharpe_series is not None and not sharpe_series.empty:
            ax.plot(sharpe_series.index, sharpe_series, 
                   linewidth=2.5, label=name, color=color)
    
    ax.axhline(0, color='#333333', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, weight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, weight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.4)
    
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    return fig


def plot_regime_timeline_annotated(predictions, sector_returns, cumulative_returns,
                                   regime_labels=None, title="Regime Timeline with Events"):
    """
    Plot 5: Regime Timeline Annotated with Real Events.
    """
    if regime_labels is None:
        regime_labels = {0: 'Risk-On', 1: 'Risk-Off', 2: 'Reflation'}
    
    # Major events to annotate
    events = [
        {'date': '2020-03-11', 'text': 'COVID Crash'},
        {'date': '2022-03-16', 'text': 'First Rate Hike'},
        {'date': '2023-03-10', 'text': 'SVB Collapse'},
    ]
    
    events_filtered = [e for e in events 
                      if pd.to_datetime(e['date']) >= cumulative_returns.index.min() 
                      and pd.to_datetime(e['date']) <= cumulative_returns.index.max()]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [1.5, 1]})
    plt.subplots_adjust(hspace=0.15)
    
    # --- Top: Cumulative Returns ---
    hmm_cols = [col for col in cumulative_returns.columns if 'HMM' in col]
    if hmm_cols:
        ax1.plot(cumulative_returns.index, cumulative_returns[hmm_cols[0]], 
                label='HMM Strategy', linewidth=3, color=QUANT_COLORS['hmm'])
    
    benchmark_cols = [col for col in cumulative_returns.columns if 'Weight' in col or 'Momentum' in col]
    for col in benchmark_cols[:2]:
        if 'SPY' in col:
            color = QUANT_COLORS['spy']
        elif 'Momentum' in col:
            color = QUANT_COLORS['momentum']
        else:
            color = QUANT_COLORS['equal']
        ax1.plot(cumulative_returns.index, cumulative_returns[col], 
                label=col, linewidth=2, linestyle='--', color=color)
    
    # Annotate events
    for event in events_filtered:
        date = pd.to_datetime(event['date'])
        ax1.axvline(x=date, color='#333333', linestyle='--', alpha=0.6)
        ax2.axvline(x=date, color='#333333', linestyle='--', alpha=0.4)
        ax1.text(date, cumulative_returns.max().max() * 0.92, event['text'], 
                rotation=0, color='#333333', fontsize=11, ha='center', weight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
    
    ax1.set_ylabel('Cumulative Return', fontsize=12, weight='bold')
    ax1.set_title(title, fontsize=16, weight='bold', pad=20)
    ax1.legend(loc='upper left', frameon=True, fontsize=10)
    ax1.grid(True, alpha=0.4)
    
    # --- Bottom: Regime Timeline ---
    dates = predictions.index
    regimes = predictions.values
    regime_colors = [QUANT_COLORS['risk_on'], QUANT_COLORS['risk_off'], QUANT_COLORS['reflation']]
    
    current_regime = regimes[0]
    start_idx = 0
    for i in range(1, len(regimes)):
        if regimes[i] != current_regime:
            color = regime_colors[current_regime]
            ax2.axvspan(dates[start_idx], dates[i-1], color=color, alpha=0.3, lw=0)
            current_regime = regimes[i]
            start_idx = i
    color = regime_colors[current_regime]
    ax2.axvspan(dates[start_idx], dates[-1], color=color, alpha=0.3, lw=0)
    
    ax2.set_ylabel('Regime', fontsize=12, weight='bold')
    ax2.set_xlabel('Date', fontsize=12, weight='bold')
    ax2.set_yticks([])
    
    # Custom legend for regimes
    patches = [mpatches.Patch(color=regime_colors[0], alpha=0.3, label='Risk-On'),
               mpatches.Patch(color=regime_colors[1], alpha=0.3, label='Risk-Off'),
               mpatches.Patch(color=regime_colors[2], alpha=0.3, label='Reflation')]
    ax2.legend(handles=patches, loc='center left', frameon=False, fontsize=10, ncol=3, bbox_to_anchor=(0.02, 0.5))
    
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def run_evaluation(backtest_results, predictions, regime_performance, save_plots=True):
    """
    Run complete evaluation and generate all metrics and visualizations.
    """
    print("\n" + "="*60)
    print("EVALUATION: PERFORMANCE ANALYSIS")
    print("="*60)
    
    results = backtest_results
    strategy_results = results['results']
    
    print("\n[1/4] Creating comparison table...")
    metrics_df, metrics_display = create_comparison_table(strategy_results)
    print("\n" + metrics_display.to_string(index=False))
    
    print("\n[2/4] Computing regime-conditional Sharpe...")
    hmm_returns = strategy_results.get('HMM Regime', {}).get('returns', pd.Series())
    if not hmm_returns.empty and predictions is not None:
        regime_sharpe = compute_regime_conditional_sharpe(hmm_returns, predictions)
        print("\nRegime-Conditional Sharpe:")
        print(regime_sharpe.round(3).to_string(index=False))
    else:
        regime_sharpe = None
    
    print("\n[3/4] Computing rolling Sharpe ratios...")
    rolling_sharpe_dict = {}
    for name, result in strategy_results.items():
        returns = result.get('returns', pd.Series())
        if not returns.empty and len(returns) > 36:
            rolling_sharpe_dict[name] = compute_rolling_sharpe(returns)
    
    print("\n[4/4] Computing turnover...")
    turnover_dict = {}
    for name, result in strategy_results.items():
        weights = result.get('weights', pd.DataFrame())
        if not weights.empty:
            turnover_dict[name] = compute_turnover(weights)
    
    print("\nTurnover (average monthly):")
    for name, turnover in turnover_dict.items():
        if not np.isnan(turnover):
            print(f"  {name}: {turnover:.2%}")
    
    if save_plots:
        print("\nGenerating visualizations...")
        
        # Plot 1: Hero Chart
        try:
            fig1 = plot_hero_chart(
                results['cumulative'],
                predictions,
                annotations=[
                    {'date': '2020-03-11', 'text': 'COVID'},
                    {'date': '2022-03-16', 'text': 'Rate Hike'},
                    {'date': '2023-03-10', 'text': 'SVB'}
                ]
            )
            fig1.savefig(OUTPUT_DIR / "hero_chart.png", dpi=300, bbox_inches='tight')
            print(f"  Saved: {OUTPUT_DIR}/hero_chart.png")
        except Exception as e:
            print(f"  WARNING: Could not generate hero chart: {e}")
        
        # Plot 2: Regime Heatmap
        try:
            if regime_performance is not None and not regime_performance.empty:
                fig2 = plot_regime_heatmap(regime_performance)
                fig2.savefig(OUTPUT_DIR / "regime_heatmap.png", dpi=300, bbox_inches='tight')
                print(f"  Saved: {OUTPUT_DIR}/regime_heatmap.png")
        except Exception as e:
            print(f"  WARNING: Could not generate regime heatmap: {e}")
        
        # Plot 3: Drawdown Comparison
        try:
            fig3 = plot_drawdown_comparison(results['drawdowns'])
            fig3.savefig(OUTPUT_DIR / "drawdowns.png", dpi=300, bbox_inches='tight')
            print(f"  Saved: {OUTPUT_DIR}/drawdowns.png")
        except Exception as e:
            print(f"  WARNING: Could not generate drawdown chart: {e}")
        
        # Plot 4: Rolling Sharpe
        try:
            if rolling_sharpe_dict:
                fig4 = plot_rolling_sharpe(rolling_sharpe_dict)
                fig4.savefig(OUTPUT_DIR / "rolling_sharpe.png", dpi=300, bbox_inches='tight')
                print(f"  Saved: {OUTPUT_DIR}/rolling_sharpe.png")
        except Exception as e:
            print(f"  WARNING: Could not generate rolling sharpe chart: {e}")
        
        # Plot 5: Annotated Timeline
        try:
            fig5 = plot_regime_timeline_annotated(
                predictions,
                results.get('hmm_strategy', {}).get('returns', pd.Series()),
                results['cumulative']
            )
            fig5.savefig(OUTPUT_DIR / "regime_timeline_events.png", dpi=300, bbox_inches='tight')
            print(f"  Saved: {OUTPUT_DIR}/regime_timeline_events.png")
        except Exception as e:
            print(f"  WARNING: Could not generate regime timeline: {e}")
        
        plt.close('all')
    
    return {
        'metrics_df': metrics_df,
        'metrics_display': metrics_display,
        'regime_sharpe': regime_sharpe,
        'rolling_sharpe': rolling_sharpe_dict,
        'turnover': turnover_dict
    }
# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing evaluation module...")
    
    # Load data and run backtest first
    from data import get_regime_data
    from models import walk_forward_predictions, sector_performance_by_regime
    from backtest import run_full_backtest
    
    # Get data
    data = get_regime_data()
    features = data['features']
    sector_returns = data['sector_returns']
    available_mask = data['available_mask']
    
    # Run models
    hmm_results = walk_forward_predictions(
        features, model_type='hmm', min_train_years=5, refit_years=1
    )
    hmm_predictions = hmm_results['predictions']
    
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
        save_plots=False
    )
    
    # Run evaluation
    eval_results = run_evaluation(
        backtest_results,
        hmm_predictions,
        hmm_performance,
        save_plots=True
    )
    
    print("\n✅ evaluation.py test complete!")