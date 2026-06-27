# models.py
"""
Regime detection models for macro-regime-rotation.
Implements GMM (baseline) and HMM (advanced) with walk-forward validation.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    N_REGIMES, RANDOM_STATE, MIN_TRAIN_YEARS, REFIT_FREQUENCY_YEARS,
    FEATURE_NAMES, DATA_DIR
)


# ============================================================================
# MODEL TRAINING
# ============================================================================

def fit_gmm(features, n_regimes=N_REGIMES, random_state=RANDOM_STATE):
    """
    Train a Gaussian Mixture Model on features.
    
    Parameters:
    -----------
    features : pd.DataFrame
        Standardized features (training data)
    n_regimes : int
        Number of regimes to detect
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    GaussianMixture : Fitted GMM model
    dict : Training metadata
    """
    print(f"Fitting GMM with {n_regimes} regimes...")
    
    # Convert to numpy array
    X = features.values
    
    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type='full',
        random_state=random_state,
        max_iter=1000,
        n_init=10
    )
    gmm.fit(X)
    
    # Get predictions
    regimes = gmm.predict(X)
    probabilities = gmm.predict_proba(X)
    
    print(f"GMM fitted. Regime distribution: {dict(pd.Series(regimes).value_counts().sort_index())}")
    
    return gmm, {
        'regimes': regimes,
        'probabilities': probabilities,
        'n_features': X.shape[1],
        'n_samples': X.shape[0]
    }


def fit_hmm(features, n_regimes=N_REGIMES, random_state=RANDOM_STATE):
    """
    Train a Hidden Markov Model with Gaussian emissions on features.
    
    Parameters:
    -----------
    features : pd.DataFrame
        Standardized features (training data)
    n_regimes : int
        Number of regimes to detect
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    hmm.GaussianHMM : Fitted HMM model
    dict : Training metadata
    """
    print(f"Fitting HMM with {n_regimes} regimes...")
    
    # Convert to numpy array
    X = features.values
    
    # Fit HMM with strong diagonal prior (forces persistence)
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type='full',
        random_state=random_state,
        n_iter=1000,
        tol=1e-4,
        init_params='stmc',  # Initialize means, transitions, covariances
        transmat_prior=10.0  # <--- NEW: Forces the model to stay in regimes longer
    )
    model.fit(X)
    
    # Get predictions (Viterbi algorithm - most likely sequence)
    regimes = model.predict(X)
    
    # Get posterior probabilities
    probabilities = model.predict_proba(X)
    
    print(f"HMM fitted. Regime distribution: {dict(pd.Series(regimes).value_counts().sort_index())}")
    print(f"  Transition matrix (row = current state, col = next state):")
    print(pd.DataFrame(model.transmat_, 
                       columns=[f'State {i}' for i in range(n_regimes)],
                       index=[f'State {i}' for i in range(n_regimes)]))
    
    return model, {
        'regimes': regimes,
        'probabilities': probabilities,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'transition_matrix': model.transmat_,
        'means': model.means_
    }
# ============================================================================
# REGIME PREDICTION
# ============================================================================

def predict_regime(model, features, model_type='hmm'):
    """
    Predict regime for new features using trained model.
    
    Parameters:
    -----------
    model : GMM or HMM
        Trained model
    features : pd.DataFrame
        Features to predict (standardized)
    model_type : str
        'hmm' or 'gmm'
    
    Returns:
    --------
    tuple : (regimes, probabilities)
    """
    X = features.values
    
    if model_type == 'hmm':
        regimes = model.predict(X)
        probabilities = model.predict_proba(X)
    else:  # gmm
        regimes = model.predict(X)
        probabilities = model.predict_proba(X)
    
    return regimes, probabilities


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

def walk_forward_predictions(features, model_type='hmm', 
                             min_train_years=MIN_TRAIN_YEARS,
                             refit_years=REFIT_FREQUENCY_YEARS,
                             n_regimes=N_REGIMES):
    """
    Run walk-forward validation with annual refitting.
    
    Parameters:
    -----------
    features : pd.DataFrame
        Full feature set with datetime index
    model_type : str
        'hmm' or 'gmm'
    min_train_years : int
        Minimum years for initial training
    refit_years : int
        Refit frequency in years
    n_regimes : int
        Number of regimes
    
    Returns:
    --------
    dict : {
        'predictions': pd.Series,  # Regime predictions at each date
        'probabilities': pd.DataFrame,  # Regime probabilities
        'models': list,  # Trained models at each refit
        'refit_dates': list  # Dates when models were refit
    }
    """
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION: {model_type.upper()}")
    print(f"{'='*60}")
    
    dates = features.index
    n_months = len(dates)
    
    # Convert years to months
    min_train_months = min_train_years * 12
    refit_months = refit_years * 12
    
    # Storage
    all_predictions = pd.Series(index=dates, dtype=int)
    all_probabilities = pd.DataFrame(index=dates, 
                                     columns=[f'State_{i}' for i in range(n_regimes)])
    models = []
    refit_dates = []
    
    # Walk forward
    train_start_idx = 0
    train_end_idx = min_train_months
    
    print(f"\nTraining initial model on {min_train_years} years ({min_train_months} months)")
    print(f"Data from {dates[train_start_idx].strftime('%Y-%m-%d')} to {dates[train_end_idx].strftime('%Y-%m-%d')}")
    
    # Progress bar
    pbar = tqdm(total=n_months - min_train_months, desc="Walk-forward")
    
    current_refit_idx = train_end_idx
    
    while current_refit_idx < n_months:
        # Get training data
        train_features = features.iloc[:current_refit_idx]
        
        if len(train_features) < min_train_months:
            current_refit_idx += 1
            pbar.update(1)
            continue
        
        # Standardize training data
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features.values)
        
        # Train model
        if model_type == 'hmm':
            model, _ = fit_hmm(pd.DataFrame(train_scaled, 
                                           columns=features.columns,
                                           index=train_features.index),
                              n_regimes=n_regimes)
        else:
            model, _ = fit_gmm(pd.DataFrame(train_scaled,
                                           columns=features.columns,
                                           index=train_features.index),
                              n_regimes=n_regimes)
        
        models.append(model)
        refit_dates.append(dates[current_refit_idx - 1])
        
        # Predict for next refit_period months (or until end)
        test_end_idx = min(current_refit_idx + refit_months, n_months)
        test_dates = dates[current_refit_idx:test_end_idx]
        
        if len(test_dates) > 0:
            # Get test features
            test_features = features.loc[test_dates]
            test_scaled = scaler.transform(test_features.values)
            
            # Predict
            if model_type == 'hmm':
                regimes = model.predict(test_scaled)
                probs = model.predict_proba(test_scaled)
            else:
                regimes = model.predict(test_scaled)
                probs = model.predict_proba(test_scaled)
            
            # Store predictions
            for i, date in enumerate(test_dates):
                all_predictions.loc[date] = regimes[i]
                all_probabilities.loc[date] = probs[i]
        
        # Move forward
        current_refit_idx = test_end_idx
        pbar.update(len(test_dates))
    
    pbar.close()
    
    # Drop any NaN predictions (shouldn't happen)
    all_predictions = all_predictions.dropna().astype(int)
    all_probabilities = all_probabilities.loc[all_predictions.index]
    
    print(f"\n✅ Walk-forward complete!")
    print(f"  {len(all_predictions)} predictions")
    print(f"  Regime distribution: {dict(all_predictions.value_counts().sort_index())}")
    print(f"  {len(models)} models trained (refit at {len(refit_dates)} dates)")
    
    # Add regime labels as a column
    results = {
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'models': models,
        'refit_dates': refit_dates
    }
    
    return results


# ============================================================================
# REGIME CHARACTERIZATION
# ============================================================================

def characterize_regimes(features, predictions):
    """
    Characterize each regime by the average feature values.
    Used to label regimes (e.g., Risk-On, Risk-Off, Reflation).
    
    Parameters:
    -----------
    features : pd.DataFrame
        Features (standardized)
    predictions : pd.Series
        Regime predictions
    
    Returns:
    --------
    pd.DataFrame : Feature means for each regime
    dict : Regime labels
    """
    # Align features and predictions
    common_idx = features.index.intersection(predictions.index)
    features_aligned = features.loc[common_idx]
    preds_aligned = predictions.loc[common_idx]
    
    # Group by regime
    regime_means = features_aligned.groupby(preds_aligned).mean()
    
    print("\nRegime Characterization:")
    print("=" * 60)
    print(features_aligned.groupby(preds_aligned).describe().round(2))
    
    # Simple labeling heuristic
    labels = {}
    for regime in regime_means.index:
        row = regime_means.loc[regime]
        
        # Risk-On: Low VIX, steep curve, tight credit
        if row.get('vix_level', 0) < 0 and row.get('yield_curve_slope', 0) > 0 and row.get('credit_spread_proxy', 0) < 0:
            labels[regime] = "Risk-On / Expansion"
        # Risk-Off: High VIX, flat/inverted curve, wide credit
        elif row.get('vix_level', 0) > 0 and row.get('yield_curve_slope', 0) < 0 and row.get('credit_spread_proxy', 0) > 0:
            labels[regime] = "Risk-Off / Recessionary"
        # Reflation: High breakeven, rising commodities, gold
        elif row.get('breakeven_proxy', 0) > 0 and row.get('oil_momentum_3m', 0) > 0 and row.get('gold_momentum_3m', 0) > 0:
            labels[regime] = "Reflation / Inflationary"
        else:
            labels[regime] = f"Regime {regime} (Mixed)"
    
    print("\nRegime Labels:")
    for regime, label in labels.items():
        print(f"  State {regime}: {label}")
    
    return regime_means, labels


# ============================================================================
# SECTOR PERFORMANCE BY REGIME
# ============================================================================

def sector_performance_by_regime(sector_returns, predictions):
    """
    Calculate average sector returns by regime.
    Used to determine which sectors to rotate into for each regime.
    
    Parameters:
    -----------
    sector_returns : pd.DataFrame
        Monthly sector returns
    predictions : pd.Series
        Regime predictions
    
    Returns:
    --------
    pd.DataFrame : Average returns by regime and sector
    """
    # Align
    common_idx = sector_returns.index.intersection(predictions.index)
    returns_aligned = sector_returns.loc[common_idx]
    preds_aligned = predictions.loc[common_idx]
    
    # Group by regime and calculate mean returns
    regime_performance = returns_aligned.groupby(preds_aligned).mean()
    
    print("\nSector Performance by Regime:")
    print("=" * 60)
    print(regime_performance.round(4))
    
    # Top sectors per regime
    for regime in regime_performance.index:
        top_sectors = regime_performance.loc[regime].nlargest(3)
        print(f"\nState {regime} top sectors:")
        for sector, ret in top_sectors.items():
            print(f"  {sector}: {ret:.4f}")
    
    return regime_performance


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(gmm_predictions, hmm_predictions, sector_returns):
    """
    Compare GMM vs HMM predictions and performance.
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON: GMM vs HMM")
    print("="*60)
    
    # Align
    common_idx = gmm_predictions.index.intersection(hmm_predictions.index)
    gmm_aligned = gmm_predictions.loc[common_idx]
    hmm_aligned = hmm_predictions.loc[common_idx]
    
    # Agreement rate
    agreement = (gmm_aligned == hmm_aligned).mean()
    print(f"\nAgreement rate: {agreement:.2%}")
    
    # Check if HMM smooths more (fewer transitions)
    gmm_transitions = (gmm_aligned.diff() != 0).sum()
    hmm_transitions = (hmm_aligned.diff() != 0).sum()
    print(f"Number of regime transitions:")
    print(f"  GMM: {gmm_transitions}")
    print(f"  HMM: {hmm_transitions}")
    print(f"  HMM smoother by: {gmm_transitions - hmm_transitions} transitions")
    
    # Compare sector returns by regime
    print("\nSector return correlation by regime:")
    print("  (Higher = more consistent sector performance patterns)")
    
    gmm_perf = sector_returns.loc[common_idx].groupby(gmm_aligned).mean()
    hmm_perf = sector_returns.loc[common_idx].groupby(hmm_aligned).mean()
    
    # Check if same number of regimes
    if len(gmm_perf) == len(hmm_perf):
        # Compare top sectors by regime
        gmm_top = gmm_perf.apply(lambda x: x.nlargest(3).index.tolist(), axis=1)
        hmm_top = hmm_perf.apply(lambda x: x.nlargest(3).index.tolist(), axis=1)
        
        # Calculate overlap
        overlaps = []
        for g_state, h_state in zip(gmm_top.index, hmm_top.index):
            overlap = set(gmm_top[g_state]) & set(hmm_top[h_state])
            overlaps.append(len(overlap) / 3)
        
        avg_overlap = np.mean(overlaps)
        print(f"  Average top-3 sector overlap: {avg_overlap:.2%}")
    
    return {
        'agreement': agreement,
        'gmm_transitions': gmm_transitions,
        'hmm_transitions': hmm_transitions
    }


# ============================================================================
# MAIN ENTRY POINT (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Testing models module...")
    
    # Load data
    from data import get_regime_data
    data = get_regime_data()
    features = data['features']
    sector_returns = data['sector_returns']
    
    # Run walk-forward for GMM
    gmm_results = walk_forward_predictions(
        features, 
        model_type='gmm',
        min_train_years=5,
        refit_years=1,
        n_regimes=N_REGIMES
    )
    
    # Run walk-forward for HMM
    hmm_results = walk_forward_predictions(
        features,
        model_type='hmm',
        min_train_years=5,
        refit_years=1,
        n_regimes=N_REGIMES
    )
    
    # Characterize regimes (using HMM)
    regime_means, labels = characterize_regimes(
        features.loc[hmm_results['predictions'].index],
        hmm_results['predictions']
    )
    
    # Sector performance by regime
    regime_performance = sector_performance_by_regime(
        sector_returns,
        hmm_results['predictions']
    )
    
    # Compare models
    comparison = compare_models(
        gmm_results['predictions'],
        hmm_results['predictions'],
        sector_returns
    )
    
    print("\n✅ models.py test complete!")