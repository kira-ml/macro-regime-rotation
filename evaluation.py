# evaluation.py
"""
Evaluation module for macro-regime-rotation strategy.
Implements core performance metrics, regime-conditional analysis,
and LinkedIn-ready visualizations.

Follows the principle: 6 well-chosen metrics > 20 random ones.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from config import OUTPUT_DIR


# ============================================================================
# TIER 1: CORE METRICS (Non-Negotiable)
# ============================================================================

def compute_metrics(returns_series, risk_free_rate=0.03, annualization=12):
    """
    Compute standard performance metrics for strategy evaluation.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Monthly strategy returns (decimal, not %)
    risk_free_rate : float
        Annual risk-free rate (default 3%)
    annualization : int
        Periods per year (12 for monthly)
    
    Returns:
    --------
    dict : Core performance metrics
    """
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
    
    # 3. Sharpe Ratio (The most important metric in quant finance)
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
    """
    Compute Sharpe ratio conditional on each regime.
    This directly tests whether the regime model adds value.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns (same index as regime_labels)
    regime_labels : pd.Series
        Regime assignments (same index as returns)
    risk_free_rate : float
        Annual risk-free rate (default 3%)
    
    Returns:
    --------
    pd.DataFrame : Sharpe per regime with metadata
    """
    monthly_rf = (1 + risk_free_rate) ** (1 / 12) - 1
    excess_returns = returns - monthly_rf
    
    results = []
    for regime in sorted(regime_labels.unique()):
        mask = regime_labels == regime
        regime_excess = excess_returns[mask]
        
        if len(regime_excess) >= 12:  # Minimum 12 months for meaningful Sharpe
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
    """
    Compute rolling Sharpe ratio over a rolling window.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    window_years : int
        Rolling window in years (default 3)
    risk_free_rate : float
        Annual risk-free rate
    
    Returns:
    --------
    pd.Series : Rolling Sharpe ratios
    """
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
    """
    Compute average monthly turnover from a history of portfolio weights.
    
    Parameters:
    -----------
    weights_history : pd.DataFrame
        Rows=dates, columns=sectors, values=weights
    
    Returns:
    --------
    float : Average monthly turnover (one-sided)
    """
    if weights_history.empty or len(weights_history) < 2:
        return np.nan
    
    weight_changes = weights_history.diff().abs().sum(axis=1)
    avg_turnover = weight_changes.mean()
    
    return avg_turnover


# ============================================================================
# TIER 3: COMPARISON TABLE
# ============================================================================

def create_comparison_table(strategy_results, risk_free_rate=0.03):
    """
    Create the single most important table: strategy comparison.
    
    Parameters:
    -----------
    strategy_results : dict
        Either:
        - {'strategy_name': pd.Series of returns, ...}
        - {'strategy_name': {'returns': pd.Series, ...}, ...} (from backtest)
    risk_free_rate : float
        Annual risk-free rate
    
    Returns:
    --------
    pd.DataFrame : Formatted comparison table
    """
    rows = []
    
    for name, result in strategy_results.items():
        # Handle both formats
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
    
    # Reorder and format
    columns = ['Strategy', 'annual_return', 'annual_volatility', 'sharpe_ratio',
               'max_drawdown', 'calmar_ratio', 'win_rate', 'cumulative_return', 'n_months']
    
    df = df[[c for c in columns if c in df.columns]]
    
    # Format for display
    df_display = df.copy()
    for col in ['annual_return', 'annual_volatility', 'max_drawdown', 
                'calmar_ratio', 'win_rate', 'cumulative_return']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A"
            )
    
    for col in ['sharpe_ratio']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"
            )
    
    return df, df_display


# ============================================================================
# VISUALIZATIONS (LinkedIn-Ready)
# ============================================================================
def plot_hero_chart(cumulative_returns, predictions, regime_labels=None,
                    annotations=None, title="Regime Rotation Strategy"):
    """
    Plot 1: The Hero Chart — Cumulative returns with regime background.
    
    This is THE LinkedIn hero image. It tells the entire story in one chart.
    
    Parameters:
    -----------
    cumulative_returns : pd.DataFrame
        Cumulative returns for multiple strategies
    predictions : pd.Series
        Regime predictions over time
    regime_labels : dict
        {regime: label} mapping
    annotations : list of dict
        [{'date': '2020-03-01', 'text': 'COVID Crash'}, ...]
    title : str
        Chart title
    """
    if regime_labels is None:
        regime_labels = {0: 'Risk-On', 1: 'Risk-Off', 2: 'Reflation'}
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.60, 0.40],
        subplot_titles=("Cumulative Returns", "Regime Timeline")
    )
    
    # Top panel: Cumulative returns
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, col in enumerate(cumulative_returns.columns):
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[col],
                mode='lines',
                name=col,
                line=dict(color=colors[i % len(colors)], width=2.5),
                hovertemplate='%{x|%b %Y}<br>%{y:.1%}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Bottom panel: Regime timeline (stacked area chart)
    regimes_unique = sorted(predictions.unique())
    regime_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Create regime probability matrix (1 for predicted regime)
    regime_probs = pd.DataFrame(index=predictions.index)
    for regime in regimes_unique:
        regime_probs[f'Regime {regime}'] = (predictions == regime).astype(float)
    
    # Stacked area chart
    for i, regime in enumerate(regimes_unique):
        label = regime_labels.get(regime, f'Regime {regime}')
        fig.add_trace(
            go.Scatter(
                x=regime_probs.index,
                y=regime_probs[f'Regime {regime}'],
                mode='lines',
                name=label,
                fill='tonexty' if i > 0 else None,
                line=dict(width=0.5, color=regime_colors[i % len(regime_colors)]),
                fillcolor=regime_colors[i % len(regime_colors)],
                opacity=0.7,
                hovertemplate='%{x|%b %Y}<br>%{y:.0%}<extra>%{fullData.name}</extra>'
            ),
            row=2, col=1
        )
    
    # Add annotations for major events
    if annotations:
        # Get y_max for annotation positioning
        y_max = cumulative_returns.max().max()
        
        for ann in annotations:
            # FIX: Convert to string to avoid Timestamp serialization issues
            date_str = ann['date']
            text = ann['text']
            
            # Convert to datetime for positioning
            date_dt = pd.to_datetime(date_str)
            
            # Add annotation as string (not Timestamp)
            fig.add_annotation(
                x=date_str,  # Pass as string, not Timestamp
                y=y_max * 0.95,
                text=text,
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowcolor='red',
                row=1, col=1,
                font=dict(size=10, color='red', weight='bold')
            )
            
            # Add vertical line using string
            fig.add_vline(
                x=date_str,  # Pass as string, not Timestamp
                line_dash="dash",
                line_color="red",
                opacity=0.3,
                row=1, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        height=700,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    fig.update_yaxes(title_text="Cumulative Return", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Regime Probability", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    return fig


def plot_regime_heatmap(regime_performance, title="Sector Performance by Regime"):
    """
    Plot 2: Regime Characterization Heatmap.
    
    Shows which sectors perform best in each regime.
    """
    # Clean data
    df = regime_performance.copy()
    
    # Calculate relative performance (within each regime)
    df_normalized = df.div(df.sum(axis=1), axis=0) * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=df_normalized.values,
        x=df.columns,
        y=[f"Regime {i}" for i in df.index],
        colorscale='RdYlGn',
        zmid=df_normalized.values.mean(),
        text=df.round(4).values,
        texttemplate='%{text:.2%}',
        textfont=dict(size=10),
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="Sector",
        yaxis_title="Regime",
        height=400,
        template='plotly_white',
        margin=dict(l=80, r=40, t=60, b=60)
    )
    
    return fig


def plot_drawdown_comparison(drawdowns, title="Drawdown Comparison"):
    """
    Plot 3: Underwater Plot — Drawdown comparison.
    """
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, col in enumerate(drawdowns.columns):
        # Get color
        color = colors[i % len(colors)]
        
        # Convert hex to RGB for fill color
        hex_color = color.lstrip('#')
        r, g, b = tuple(int(hex_color[j:j+2], 16) for j in (0, 2, 4))
        
        fig.add_trace(go.Scatter(
            x=drawdowns.index,
            y=drawdowns[col],
            mode='lines',
            name=col,
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=f'rgba({r},{g},{b},0.1)',
            hovertemplate='%{x|%b %Y}<br>%{y:.1%}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        height=400,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    return fig


def plot_rolling_sharpe(rolling_sharpe_dict, title="Rolling 3-Year Sharpe Ratio"):
    """
    Plot 5: Rolling Sharpe Ratio.
    """
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, sharpe_series) in enumerate(rolling_sharpe_dict.items()):
        if sharpe_series is not None and not sharpe_series.empty:
            fig.add_trace(go.Scatter(
                x=sharpe_series.index,
                y=sharpe_series,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate='%{x|%b %Y}<br>Sharpe: %{y:.2f}<extra></extra>'
            ))
    
    # Add horizontal line at Sharpe = 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        height=400,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    return fig


def plot_regime_timeline_annotated(predictions, sector_returns, cumulative_returns,
                                   regime_labels=None, title="Regime Timeline with Events"):
    """
    Plot 4: Regime Timeline Annotated with Real Events.
    Validates that the model found real economic regimes.
    """
    if regime_labels is None:
        regime_labels = {0: 'Risk-On', 1: 'Risk-Off', 2: 'Reflation'}
    
    # Major events to annotate (adjust based on your data range)
    events = [
        {'date': '2008-09-15', 'text': 'Lehman Collapse'},
        {'date': '2011-08-05', 'text': 'US Debt Downgrade'},
        {'date': '2016-02-11', 'text': 'Oil Crash Bottom'},
        {'date': '2018-12-24', 'text': 'Volmageddon'},
        {'date': '2020-03-11', 'text': 'COVID Crash'},
        {'date': '2022-03-16', 'text': 'First Rate Hike'},
        {'date': '2023-03-10', 'text': 'SVB Collapse'},
    ]
    
    # Filter events to date range
    events_filtered = [e for e in events 
                      if pd.to_datetime(e['date']) >= cumulative_returns.index.min() 
                      and pd.to_datetime(e['date']) <= cumulative_returns.index.max()]
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.50, 0.50],
        subplot_titles=("Cumulative Returns", "Regime Timeline with Events")
    )
    
    # Top: Cumulative returns (only HMM strategy)
    hmm_cols = [col for col in cumulative_returns.columns if 'HMM' in col]
    if hmm_cols:
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[hmm_cols[0]],
                mode='lines',
                name='HMM Strategy',
                line=dict(color='#1f77b4', width=2.5)
            ),
            row=1, col=1
        )
    
    # Add benchmark for comparison
    benchmark_cols = [col for col in cumulative_returns.columns if 'Weight' in col or 'Momentum' in col]
    for col in benchmark_cols[:2]:
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[col],
                mode='lines',
                name=col,
                line=dict(dash='dash', width=1.5)
            ),
            row=1, col=1
        )
    
    # Bottom: Regime timeline as colored segments
    dates = predictions.index
    regimes = predictions.values
    regime_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
    
    # Add regime segments
    current_regime = regimes[0]
    start_idx = 0
    
    for i in range(1, len(regimes)):
        if regimes[i] != current_regime:
            color = regime_colors.get(current_regime, '#808080')
            label = regime_labels.get(current_regime, f'State {current_regime}')
            
            fig.add_vrect(
                x0=dates[start_idx],
                x1=dates[i-1],
                fillcolor=color,
                opacity=0.3,
                line_width=0,
                row=2, col=1,
                annotation_text=label,
                annotation_position="top left"
            )
            
            current_regime = regimes[i]
            start_idx = i
    
    # Add last segment
    color = regime_colors.get(current_regime, '#808080')
    label = regime_labels.get(current_regime, f'State {current_regime}')
    fig.add_vrect(
        x0=dates[start_idx],
        x1=dates[-1],
        fillcolor=color,
        opacity=0.3,
        line_width=0,
        row=2, col=1,
        annotation_text=label,
        annotation_position="top left"
    )
    
    # Add event annotations
    for event in events_filtered:
        date = pd.to_datetime(event['date'])
        if date >= cumulative_returns.index.min() and date <= cumulative_returns.index.max():
            fig.add_vline(
                x=date,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                row=1, col=1
            )
            fig.add_vline(
                x=date,
                line_dash="dash",
                line_color="red",
                opacity=0.3,
                row=2, col=1
            )
            
            # Add annotation on top panel
            y_max = cumulative_returns.max().max()
            fig.add_annotation(
                x=date,
                y=y_max * 0.90,
                text=event['text'],
                showarrow=True,
                arrowhead=2,
                arrowsize=1.0,
                arrowcolor='red',
                row=1, col=1,
                font=dict(size=9, color='red')
            )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        height=700,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    fig.update_yaxes(title_text="Cumulative Return", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Regime", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    return fig


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_evaluation(backtest_results, predictions, regime_performance, 
                   save_plots=True):
    """
    Run complete evaluation and generate all metrics and visualizations.
    
    Parameters:
    -----------
    backtest_results : dict
        Results from backtest.run_full_backtest()
    predictions : pd.Series
        HMM regime predictions
    regime_performance : pd.DataFrame
        Sector performance by regime
    save_plots : bool
        Whether to save plots to OUTPUT_DIR
    
    Returns:
    --------
    dict : All evaluation results
    """
    print("\n" + "="*60)
    print("EVALUATION: PERFORMANCE ANALYSIS")
    print("="*60)
    
    results = backtest_results
    strategy_results = results['results']
    
    # 1. Create comparison table
    print("\n[1/4] Creating comparison table...")
    metrics_df, metrics_display = create_comparison_table(strategy_results)
    print("\n" + metrics_display.to_string(index=False))
    
    # 2. Compute regime-conditional Sharpe
    print("\n[2/4] Computing regime-conditional Sharpe...")
    hmm_returns = strategy_results.get('HMM Regime', {}).get('returns', pd.Series())
    if not hmm_returns.empty and predictions is not None:
        regime_sharpe = compute_regime_conditional_sharpe(hmm_returns, predictions)
        print("\nRegime-Conditional Sharpe:")
        print(regime_sharpe.round(3).to_string(index=False))
    else:
        regime_sharpe = None
    
    # 3. Compute rolling Sharpe
    print("\n[3/4] Computing rolling Sharpe ratios...")
    rolling_sharpe_dict = {}
    for name, result in strategy_results.items():
        returns = result.get('returns', pd.Series())
        if not returns.empty and len(returns) > 36:
            rolling_sharpe_dict[name] = compute_rolling_sharpe(returns)
    
    # 4. Compute turnover
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
    
    # 5. Generate visualizations
    if save_plots:
        print("\nGenerating visualizations...")
        
        # Plot 1: Hero Chart
        fig1 = plot_hero_chart(
            results['cumulative'],
            predictions,
            annotations=[
                {'date': '2020-03-11', 'text': 'COVID'},
                {'date': '2022-03-16', 'text': 'Rate Hike'},
                {'date': '2023-03-10', 'text': 'SVB'}
            ]
        )
        fig1.write_html(OUTPUT_DIR / "hero_chart.html")
        fig1.write_image(OUTPUT_DIR / "hero_chart.png", scale=2)
        print(f"  Saved: {OUTPUT_DIR}/hero_chart.png")
        
        # Plot 2: Regime Heatmap
        if regime_performance is not None and not regime_performance.empty:
            fig2 = plot_regime_heatmap(regime_performance)
            fig2.write_html(OUTPUT_DIR / "regime_heatmap.html")
            fig2.write_image(OUTPUT_DIR / "regime_heatmap.png", scale=2)
            print(f"  Saved: {OUTPUT_DIR}/regime_heatmap.png")
        
        # Plot 3: Drawdown Comparison
        fig3 = plot_drawdown_comparison(results['drawdowns'])
        fig3.write_html(OUTPUT_DIR / "drawdowns.html")
        fig3.write_image(OUTPUT_DIR / "drawdowns.png", scale=2)
        print(f"  Saved: {OUTPUT_DIR}/drawdowns.png")
        
        # Plot 4: Rolling Sharpe
        if rolling_sharpe_dict:
            fig4 = plot_rolling_sharpe(rolling_sharpe_dict)
            fig4.write_html(OUTPUT_DIR / "rolling_sharpe.html")
            fig4.write_image(OUTPUT_DIR / "rolling_sharpe.png", scale=2)
            print(f"  Saved: {OUTPUT_DIR}/rolling_sharpe.png")
        
        # Plot 5: Annotated Timeline
        fig5 = plot_regime_timeline_annotated(
            predictions,
            results.get('hmm_strategy', {}).get('returns', pd.Series()),
            results['cumulative']
        )
        fig5.write_html(OUTPUT_DIR / "regime_timeline_events.html")
        fig5.write_image(OUTPUT_DIR / "regime_timeline_events.png", scale=2)
        print(f"  Saved: {OUTPUT_DIR}/regime_timeline_events.png")
    
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