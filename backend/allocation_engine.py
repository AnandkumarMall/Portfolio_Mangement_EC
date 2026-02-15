"""
Allocation engine for computing portfolio weights based on regime and volatility.
"""

import pandas as pd
import numpy as np
from typing import Dict, List

from config import REGIME_EXPOSURE


def inverse_volatility_weights(volatilities: pd.Series) -> pd.Series:
    """
    Compute inverse volatility weights.
    
    Args:
        volatilities: Series of asset volatilities
        
    Returns:
        Series of normalized weights
    """
    # Handle NaN and zero volatilities
    vols = volatilities.copy()
    vols = vols.fillna(vols.mean())  # Fill NaN with mean
    vols = vols.replace(0, vols[vols > 0].min())  # Replace zeros with min non-zero
    
    # Compute inverse volatility
    inv_vol = 1 / vols
    
    # Normalize to sum to 1
    weights = inv_vol / inv_vol.sum()
    
    return weights


def equal_weights(tickers: List[str]) -> pd.Series:
    """
    Compute equal weights for all assets.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        Series of equal weights
    """
    n = len(tickers)
    weights = pd.Series(1/n, index=tickers)
    return weights


def compute_allocation(
    regime: str,
    volatilities: pd.Series,
    method: str = 'inverse_vol',
    regime_exposure: Dict[str, float] = None
) -> pd.Series:
    """
    Compute portfolio allocation based on regime and weighting method.
    
    Args:
        regime: Current market regime ('BULL', 'VOLATILE', 'CRASH')
        volatilities: Series of asset volatilities
        method: Weighting method ('inverse_vol' or 'equal_weight')
        regime_exposure: Dictionary mapping regimes to exposure levels
        
    Returns:
        Series of portfolio weights (sum <= 1.0)
    """
    if regime_exposure is None:
        regime_exposure = REGIME_EXPOSURE
    
    # Get target exposure based on regime
    target_exposure = regime_exposure.get(regime, 0.6)  # Default to 60% if unknown
    
    # Compute base weights
    if method == 'inverse_vol':
        base_weights = inverse_volatility_weights(volatilities)
    elif method == 'equal_weight':
        base_weights = equal_weights(volatilities.index.tolist())
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Scale weights by target exposure
    final_weights = base_weights * target_exposure
    
    return final_weights


def rebalance_portfolio(
    current_weights: pd.Series,
    target_weights: pd.Series,
    threshold: float = 0.05
) -> tuple:
    """
    Determine if rebalancing is needed and compute trades.
    
    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        threshold: Minimum deviation to trigger rebalance
        
    Returns:
        Tuple of (need_rebalance: bool, trades: Series)
    """
    # Align indices
    all_tickers = current_weights.index.union(target_weights.index)
    current = current_weights.reindex(all_tickers, fill_value=0)
    target = target_weights.reindex(all_tickers, fill_value=0)
    
    # Compute deviation
    deviation = (target - current).abs()
    max_deviation = deviation.max()
    
    need_rebalance = max_deviation > threshold
    
    # Compute trades (positive = buy, negative = sell)
    trades = target - current
    
    return need_rebalance, trades


def apply_position_limits(
    weights: pd.Series,
    max_weight: float = 0.10
) -> pd.Series:
    """
    Apply position limits to weights.
    
    Args:
        weights: Portfolio weights
        max_weight: Maximum weight per position
        
    Returns:
        Adjusted weights with limits applied
    """
    # Cap weights at max_weight
    capped_weights = weights.clip(upper=max_weight)
    
    # Check if we need to renormalize
    total_weight = capped_weights.sum()
    
    if total_weight > 1.0:
        # This shouldn't happen if we're starting from normalized weights
        # but we'll handle it anyway
        capped_weights = capped_weights / total_weight
    
    return capped_weights


def compute_turnover(current_weights: pd.Series, new_weights: pd.Series) -> float:
    """
    Compute portfolio turnover.
    
    Args:
        current_weights: Current portfolio weights
        new_weights: New portfolio weights
        
    Returns:
        Turnover as a percentage (0-100)
    """
    # Align indices
    all_tickers = current_weights.index.union(new_weights.index)
    current = current_weights.reindex(all_tickers, fill_value=0)
    new = new_weights.reindex(all_tickers, fill_value=0)
    
    # Turnover is sum of absolute changes
    turnover = (new - current).abs().sum()
    
    return turnover * 100  # Return as percentage
