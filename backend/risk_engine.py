"""
Risk management engine applying multiple layers of risk controls.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from config import RISK_PARAMS


def apply_volatility_targeting(
    weights: pd.Series,
    portfolio_volatility: float,
    returns: pd.DataFrame,
    logs: List[Dict]
) -> Tuple[pd.Series, List[Dict]]:
    """
    Scale down portfolio if volatility exceeds target.
    
    Args:
        weights: Current portfolio weights
        portfolio_volatility: Current annualized portfolio volatility
        returns: Returns DataFrame (for covariance calculation)
        logs: List to append risk events
        
    Returns:
        Tuple of (adjusted weights, updated logs)
    """
    target_vol = RISK_PARAMS['target_volatility']
    adjustment_vol = RISK_PARAMS['volatility_adjustment']
    
    if portfolio_volatility > target_vol:
        # Scale down to adjustment level
        scale_factor = adjustment_vol / portfolio_volatility
        adjusted_weights = weights * scale_factor
        
        logs.append({
            'event_type': 'VOL_BREACH',
            'details': f'Portfolio vol {portfolio_volatility:.2%} > target {target_vol:.2%}. Scaled to {adjustment_vol:.2%}',
            'scale_factor': scale_factor
        })
        
        return adjusted_weights, logs
    
    return weights, logs


def apply_drawdown_protection(
    weights: pd.Series,
    current_drawdown: float,
    logs: List[Dict]
) -> Tuple[pd.Series, List[Dict]]:
    """
    Reduce exposure or move to cash based on drawdown level.
    
    Args:
        weights: Current portfolio weights
        current_drawdown: Current drawdown (negative value)
        logs: List to append risk events
        
    Returns:
        Tuple of (adjusted weights, updated logs)
    """
    dd_threshold_1 = RISK_PARAMS['drawdown_threshold_1']
    dd_threshold_2 = RISK_PARAMS['drawdown_threshold_2']
    
    # Convert to positive for easier comparison
    dd = abs(current_drawdown)
    
    if dd > dd_threshold_2:
        # Emergency: move to 100% cash
        adjusted_weights = weights * 0.0
        
        logs.append({
            'event_type': 'DRAWDOWN_20',
            'details': f'Drawdown {dd:.2%} > {dd_threshold_2:.2%}. Moving to 100% cash',
            'scale_factor': 0.0
        })
        
        return adjusted_weights, logs
    
    elif dd > dd_threshold_1:
        # Reduce exposure by 50%
        adjusted_weights = weights * 0.5
        
        logs.append({
            'event_type': 'DRAWDOWN_10',
            'details': f'Drawdown {dd:.2%} > {dd_threshold_1:.2%}. Reducing exposure by 50%',
            'scale_factor': 0.5
        })
        
        return adjusted_weights, logs
    
    return weights, logs


def apply_position_caps(
    weights: pd.Series,
    logs: List[Dict]
) -> Tuple[pd.Series, List[Dict]]:
    """
    Cap individual positions at maximum weight.
    
    Args:
        weights: Current portfolio weights
        logs: List to append risk events
        
    Returns:
        Tuple of (adjusted weights, updated logs)
    """
    max_weight = RISK_PARAMS['max_position_weight']
    
    # Find positions exceeding max
    overly_concentrated = weights > max_weight
    
    if overly_concentrated.any():
        # Cap at max weight
        adjusted_weights = weights.clip(upper=max_weight)
        
        # Renormalize to maintain total exposure
        total_before = weights.sum()
        total_after = adjusted_weights.sum()
        
        if total_after < total_before:
            # Scale up other positions proportionally
            scale_factor = total_before / total_after
            adjusted_weights = adjusted_weights * scale_factor
        
        capped_tickers = weights[overly_concentrated].index.tolist()
        
        logs.append({
            'event_type': 'POSITION_CAP',
            'details': f'Capped {len(capped_tickers)} positions at {max_weight:.2%}: {capped_tickers}',
            'capped_tickers': capped_tickers
        })
        
        return adjusted_weights, logs
    
    return weights, logs


def apply_stop_loss(
    weights: pd.Series,
    daily_returns: pd.Series,
    logs: List[Dict]
) -> Tuple[pd.Series, List[Dict]]:
    """
    Apply stop-loss: zero out positions with large daily losses.
    
    Args:
        weights: Current portfolio weights
        daily_returns: Most recent daily returns for each stock
        logs: List to append risk events
        
    Returns:
        Tuple of (adjusted weights, updated logs)
    """
    stop_loss_threshold = RISK_PARAMS['stop_loss_threshold']
    
    # Find stocks with returns below threshold
    stopped_out = daily_returns < stop_loss_threshold
    
    if stopped_out.any():
        # Zero out stopped positions
        adjusted_weights = weights.copy()
        stopped_tickers = daily_returns[stopped_out].index.tolist()
        
        for ticker in stopped_tickers:
            if ticker in adjusted_weights.index:
                adjusted_weights[ticker] = 0.0
        
        # Renormalize remaining positions
        if adjusted_weights.sum() > 0:
            total_exposure = weights.sum()
            adjusted_weights = adjusted_weights / adjusted_weights.sum() * total_exposure
        
        logs.append({
            'event_type': 'STOP_LOSS',
            'details': f'Stop-loss triggered for {len(stopped_tickers)} stocks: {stopped_tickers}',
            'stopped_tickers': stopped_tickers,
            'returns': daily_returns[stopped_out].to_dict()
        })
        
        return adjusted_weights, logs
    
    return weights, logs


def compute_portfolio_volatility(
    weights: pd.Series,
    returns: pd.DataFrame,
    window: int = 63
) -> float:
    """
    Compute annualized portfolio volatility.
    
    Args:
        weights: Portfolio weights
        returns: Returns DataFrame
        window: Rolling window for covariance
        
    Returns:
        Annualized portfolio volatility
    """
    # Align weights with returns columns
    weights_aligned = weights.reindex(returns.columns, fill_value=0)
    
    # Get recent returns for covariance
    recent_returns = returns.tail(window)
    
    if len(recent_returns) < window:
        # Not enough data, use available data
        recent_returns = returns
    
    # Compute covariance matrix
    cov_matrix = recent_returns.cov()
    
    # Portfolio variance
    portfolio_variance = np.dot(weights_aligned.values, np.dot(cov_matrix.values, weights_aligned.values))
    
    # Annualized volatility (assuming 252 trading days)
    portfolio_vol = np.sqrt(portfolio_variance * 252)
    
    return portfolio_vol


def apply_risk_controls(
    weights: pd.Series,
    portfolio_state: Dict,
    features: Dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    date: pd.Timestamp
) -> Tuple[pd.Series, List[Dict]]:
    """
    Apply all risk controls sequentially.
    
    Args:
        weights: Initial portfolio weights (from allocation engine)
        portfolio_state: Current portfolio state (equity_curve, etc.)
        features: Dictionary of features
        returns: Returns DataFrame
        date: Current date
        
    Returns:
        Tuple of (final adjusted weights, risk event logs)
    """
    logs = []
    adjusted_weights = weights.copy()
    
    # 1. Position caps (apply first to ensure no single position is too large)
    adjusted_weights, logs = apply_position_caps(adjusted_weights, logs)
    
    # 2. Volatility targeting
    portfolio_vol = compute_portfolio_volatility(adjusted_weights, returns.loc[:date])
    adjusted_weights, logs = apply_volatility_targeting(
        adjusted_weights, portfolio_vol, returns.loc[:date], logs
    )
    
    # 3. Drawdown protection
    if 'equity_curve' in portfolio_state and len(portfolio_state['equity_curve']) > 0:
        from feature_engineering import compute_portfolio_drawdown
        equity_series = pd.Series(portfolio_state['equity_curve'])
        current_dd = compute_portfolio_drawdown(equity_series).iloc[-1]
        adjusted_weights, logs = apply_drawdown_protection(adjusted_weights, current_dd, logs)
    
    # 4. Stop-loss (use previous day's returns)
    if date in returns.index:
        daily_rets = returns.loc[date]
        adjusted_weights, logs = apply_stop_loss(adjusted_weights, daily_rets, logs)
    
    # Add date to all log entries
    for log in logs:
        log['date'] = date
    
    return adjusted_weights, logs
