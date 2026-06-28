"""
EXPERIMENT: Regime Count Optimization
Tests 2, 3, and 4 regimes for both GMM and HMM.
"""

from data import get_regime_data
from models import walk_forward_predictions, sector_performance_by_regime
from backtest import run_full_backtest
import pandas as pd

print("="*60)
print("REGIME COUNT EXPERIMENT")
print("="*60)

# Load data
data = get_regime_data()
features = data['features']
sector_returns = data['sector_returns']
available_mask = data['available_mask']

# Store results for comparison
results_summary = []

for n in [2, 3, 4]:
    print(f"\n{'='*60}")
    print(f"TESTING N_REGIMES = {n}")
    print('='*60)
    
    # --- HMM ---
    hmm_results = walk_forward_predictions(
        features, model_type='hmm', n_regimes=n
    )
    hmm_preds = hmm_results['predictions']
    
    # --- GMM ---
    gmm_results = walk_forward_predictions(
        features, model_type='gmm', n_regimes=n
    )
    gmm_preds = gmm_results['predictions']
    
    # Get performance by regime
    hmm_perf = sector_performance_by_regime(sector_returns, hmm_preds)
    gmm_perf = sector_performance_by_regime(sector_returns, gmm_preds)
    
    # Run backtest
    bt_results = run_full_backtest(
        features, sector_returns, available_mask,
        hmm_preds, gmm_preds,
        hmm_perf, gmm_perf,
        save_plots=False
    )
    
    # Extract metrics
    metrics = bt_results['metrics']
    hmm_row = metrics[metrics['Strategy'] == 'HMM Regime'].iloc[0]
    gmm_row = metrics[metrics['Strategy'] == 'GMM Regime'].iloc[0]
    
    results_summary.append({
        'N_Regimes': n,
        'HMM_Sharpe': hmm_row['sharpe_ratio'],
        'HMM_Return': hmm_row['annual_return'],
        'HMM_Turnover': bt_results['results']['HMM Regime']['turnover'].mean(),
        'GMM_Sharpe': gmm_row['sharpe_ratio'],
        'GMM_Return': gmm_row['annual_return'],
        'GMM_Turnover': bt_results['results']['GMM Regime']['turnover'].mean(),
    })

# Print Final Summary Table
print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
summary_df = pd.DataFrame(results_summary).round(4)
print(summary_df.to_string(index=False))

# Save results for reference in README
from config import OUTPUT_DIR
summary_df.to_csv(OUTPUT_DIR / "regime_count_experiment.csv", index=False)
print(f"\nResults saved to {OUTPUT_DIR}/regime_count_experiment.csv")