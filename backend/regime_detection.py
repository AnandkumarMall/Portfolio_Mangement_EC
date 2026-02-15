"""
Regime detection module using rule-based classification.
No machine learning - purely threshold-based logic.
"""

import pandas as pd
import numpy as np
from typing import Dict

from config import REGIME_THRESHOLDS


def detect_regime_single_date(
    volatility: float,
    momentum: float,
    drawdown: float,
    thresholds: Dict[str, float] = None
) -> str:
    """
    Detect market regime for a single date based on features.
    
    Regime Logic:
    - CRASH: Drawdown exceeds threshold
    - VOLATILE: High volatility
    - BULL: Low volatility + positive momentum
    
    Args:
        volatility: Current 21-day volatility
        momentum: Current 252-day momentum
        drawdown: Current drawdown (negative value)
        thresholds: Dictionary of threshold values
        
    Returns:
        Regime label: 'BULL', 'VOLATILE', or 'CRASH'
    """
    if thresholds is None:
        thresholds = REGIME_THRESHOLDS
    
    # Handle NaN values
    if pd.isna(volatility) or pd.isna(momentum) or pd.isna(drawdown):
        return "VOLATILE"  # Default to volatile if data unavailable
    
    # Rule-based classification
    # Priority: CRASH > VOLATILE > BULL
    
    if drawdown < -thresholds['crash_drawdown']:
        return "CRASH"
    
    if volatility > thresholds['volatile_vol']:
        return "VOLATILE"
    
    if momentum > 0 and volatility < thresholds['bull_vol']:
        return "BULL"
    
    # Default to volatile
    return "VOLATILE"


def detect_regime_timeseries(
    features: Dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    equity_curve: pd.Series = None
) -> pd.Series:
    """
    Detect regime for entire time series.
    
    Args:
        features: Dictionary of feature DataFrames
        returns: Returns DataFrame (for portfolio calculation)
        equity_curve: Portfolio equity curve (if available)
        
    Returns:
        Series of regime labels with dates as index
    """
    # Get portfolio-level features
    volatility_21d = features['volatility_21d'].mean(axis=1)  # Average across stocks
    momentum_252d = features['momentum_252d'].mean(axis=1)    # Average across stocks
    
    # Compute drawdown
    if equity_curve is not None:
        from feature_engineering import compute_portfolio_drawdown
        drawdown = compute_portfolio_drawdown(equity_curve)
    else:
        # Use equal-weighted portfolio for initial regime detection
        portfolio_returns = returns.mean(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        from feature_engineering import compute_portfolio_drawdown
        drawdown = compute_portfolio_drawdown(cumulative_returns)
    
    # Detect regime for each date
    regimes = pd.Series(index=volatility_21d.index, dtype=str)
    
    for date in volatility_21d.index:
        vol = volatility_21d.loc[date]
        mom = momentum_252d.loc[date]
        dd = drawdown.loc[date]
        
        regimes.loc[date] = detect_regime_single_date(vol, mom, dd)
    
    return regimes


def get_regime_statistics(regimes: pd.Series) -> pd.DataFrame:
    """
    Compute statistics about regime distribution.
    
    Args:
        regimes: Series of regime labels
        
    Returns:
        DataFrame with regime statistics
    """
    stats = regimes.value_counts()
    stats_pct = (stats / len(regimes) * 100).round(2)
    
    result = pd.DataFrame({
        'Count': stats,
        'Percentage': stats_pct
    })
    
    return result


def detect_regime_changes(regimes: pd.Series) -> pd.DataFrame:
    """
    Detect when regime changes occur.
    
    Args:
        regimes: Series of regime labels
        
    Returns:
        DataFrame with regime change events
    """
    # Find where regime changes
    regime_changes = regimes != regimes.shift(1)
    
    change_dates = regimes[regime_changes].index[1:]  # Skip first date
    
    changes = []
    for date in change_dates:
        prev_regime = regimes.loc[regimes.index < date].iloc[-1]
        new_regime = regimes.loc[date]
        changes.append({
            'date': date,
            'from_regime': prev_regime,
            'to_regime': new_regime
        })
    
    return pd.DataFrame(changes)
