"""
Feature engineering module for computing rolling metrics.
CRITICAL: All features are lagged by 1 day using .shift(1) to prevent lookahead bias.
"""

import pandas as pd
import numpy as np
from typing import Tuple

from config import WINDOW_SHORT, WINDOW_MEDIUM, WINDOW_LONG, MA_SHORT, MA_LONG


def compute_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute rolling volatility (standard deviation).
    
    Args:
        returns: Daily returns DataFrame
        window: Rolling window size
        
    Returns:
        DataFrame of rolling volatility (lagged by 1 day)
    """
    volatility = returns.rolling(window=window).std()
    # Shift by 1 to prevent lookahead bias
    return volatility.shift(1)


def compute_momentum(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute momentum as total return over the window.
    
    Args:
        prices: Price DataFrame
        window: Lookback window
        
    Returns:
        DataFrame of momentum (lagged by 1 day)
    """
    momentum = prices.pct_change(periods=window)
    # Shift by 1 to prevent lookahead bias
    return momentum.shift(1)


def compute_moving_averages(prices: pd.DataFrame, windows: list) -> dict:
    """
    Compute multiple moving averages.
    
    Args:
        prices: Price DataFrame
        windows: List of window sizes
        
    Returns:
        Dictionary of moving average DataFrames (lagged by 1 day)
    """
    mas = {}
    for window in windows:
        ma = prices.rolling(window=window).mean()
        # Shift by 1 to prevent lookahead bias
        mas[f'MA_{window}'] = ma.shift(1)
    return mas


def compute_covariance_matrix(returns: pd.DataFrame, window: int, date: pd.Timestamp) -> pd.DataFrame:
    """
    Compute rolling covariance matrix at a specific date.
    Uses data UP TO (but not including) the given date.
    
    Args:
        returns: Daily returns DataFrame
        window: Rolling window size
        date: Date for which to compute covariance
        
    Returns:
        Covariance matrix (DataFrame)
    """
    # Get data up to (but not including) the date
    mask = returns.index < date
    recent_returns = returns[mask].tail(window)
    
    if len(recent_returns) < window:
        # Not enough data
        return pd.DataFrame(np.nan, index=returns.columns, columns=returns.columns)
    
    return recent_returns.cov()


def compute_correlation_matrix(returns: pd.DataFrame, window: int, date: pd.Timestamp) -> pd.DataFrame:
    """
    Compute rolling correlation matrix at a specific date.
    Uses data UP TO (but not including) the given date.
    
    Args:
        returns: Daily returns DataFrame
        window: Rolling window size
        date: Date for which to compute correlation
        
    Returns:
        Correlation matrix (DataFrame)
    """
    # Get data up to (but not including) the date
    mask = returns.index < date
    recent_returns = returns[mask].tail(window)
    
    if len(recent_returns) < window:
        # Not enough data
        return pd.DataFrame(np.nan, index=returns.columns, columns=returns.columns)
    
    return recent_returns.corr()


def compute_portfolio_drawdown(equity_curve: pd.Series) -> pd.Series:
    """
    Compute running maximum drawdown from equity curve.
    
    Args:
        equity_curve: Equity curve (portfolio value over time)
        
    Returns:
        Series of drawdown percentages (negative values)
    """
    # Compute running maximum
    running_max = equity_curve.expanding().max()
    
    # Compute drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    return drawdown


def compute_all_features(
    prices: pd.DataFrame, 
    returns: pd.DataFrame
) -> dict:
    """
    Compute all features required for regime detection and allocation.
    
    Args:
        prices: Price DataFrame
        returns: Returns DataFrame
        
    Returns:
        Dictionary containing all features
    """
    features = {}
    
    # Volatilities
    features['volatility_21d'] = compute_volatility(returns, WINDOW_SHORT)
    features['volatility_63d'] = compute_volatility(returns, WINDOW_MEDIUM)
    
    # Momentum
    features['momentum_252d'] = compute_momentum(prices, WINDOW_LONG)
    
    # Moving averages
    mas = compute_moving_averages(prices, [MA_SHORT, MA_LONG])
    features.update(mas)
    
    # Price relative to MA
    features['price_vs_ma50'] = (prices / mas['MA_50']) - 1
    features['price_vs_ma200'] = (prices / mas['MA_200']) - 1
    
    print(f"Computed features: {list(features.keys())}")
    
    return features


def aggregate_portfolio_features(features: dict, weights: pd.Series = None) -> pd.Series:
    """
    Aggregate stock-level features to portfolio level.
    If weights not provided, use equal weights.
    
    Args:
        features: Dictionary of feature DataFrames
        weights: Portfolio weights (Series)
        
    Returns:
        Series of portfolio-level features with same index as input
    """
    if weights is None:
        # Equal weights
        n_assets = features['volatility_21d'].shape[1]
        weights = pd.Series(1/n_assets, index=features['volatility_21d'].columns)
    
    portfolio_features = pd.DataFrame(index=features['volatility_21d'].index)
    
    # Portfolio volatility (weighted average for simplicity)
    portfolio_features['portfolio_vol'] = (features['volatility_21d'] * weights).sum(axis=1)
    
    # Portfolio momentum (weighted average)
    portfolio_features['portfolio_momentum'] = (features['momentum_252d'] * weights).sum(axis=1)
    
    return portfolio_features
