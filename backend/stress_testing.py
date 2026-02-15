"""
Stress testing module for applying market shocks and re-running backtests.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def apply_market_shock(returns: pd.DataFrame, shock_amount: float = -0.05) -> pd.DataFrame:
    """
    Apply a uniform market shock to all returns.
    
    Args:
        returns: Original returns DataFrame
        shock_amount: Amount to subtract from all returns (default: -5%)
        
    Returns:
        Stressed returns DataFrame
    """
    stressed_returns = returns + shock_amount
    return stressed_returns


def apply_volatility_spike(returns: pd.DataFrame, multiplier: float = 2.0) -> pd.DataFrame:
    """
    Multiply all returns by a factor to simulate volatility spike.
    
    Args:
        returns: Original returns DataFrame
        multiplier: Volatility multiplier (default: 2x)
        
    Returns:
        Stressed returns DataFrame
    """
    # Center returns around zero for multiplication
    mean_returns = returns.mean()
    demeaned_returns = returns - mean_returns
    
    # Multiply by factor
    stressed_demeaned = demeaned_returns * multiplier
    
    # Add mean back
    stressed_returns = stressed_demeaned + mean_returns
    
    return stressed_returns


def apply_correlation_spike(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Force all returns to move together (full correlation).
    Replace individual returns with market average.
    
    Args:
        returns: Original returns DataFrame
        
    Returns:
        Stressed returns DataFrame
    """
    # Compute market average (equal-weighted)
    market_return = returns.mean(axis=1)
    
    # Replace all stocks with market return
    stressed_returns = pd.DataFrame(
        np.tile(market_return.values.reshape(-1, 1), (1, returns.shape[1])),
        index=returns.index,
        columns=returns.columns
    )
    
    return stressed_returns


def apply_stress_scenario(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    scenario_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply stress scenario to returns and recalculate prices.
    
    Args:
        returns: Original returns DataFrame
        prices: Original prices DataFrame
        scenario_type: 'market_shock', 'volatility_spike', or 'correlation_spike'
        
    Returns:
        Tuple of (stressed_returns, stressed_prices)
    """
    if scenario_type == 'market_shock':
        stressed_returns = apply_market_shock(returns, shock_amount=-0.05)
    elif scenario_type == 'volatility_spike':
        stressed_returns = apply_volatility_spike(returns, multiplier=2.0)
    elif scenario_type == 'correlation_spike':
        stressed_returns = apply_correlation_spike(returns)
    else:
        raise ValueError(f"Unknown stress scenario: {scenario_type}")
    
    # Recalculate prices from stressed returns
    # Start with first day's actual prices
    initial_prices = prices.iloc[0]
    
    # Compute cumulative product of (1 + returns)
    stressed_prices = initial_prices * (1 + stressed_returns).cumprod()
    
    return stressed_returns, stressed_prices


def get_scenario_description(scenario_type: str) -> str:
    """
    Get human-readable description of stress scenario.
    
    Args:
        scenario_type: Scenario type
        
    Returns:
        Description string
    """
    descriptions = {
        'market_shock': 'Market Shock: -5% applied to all daily returns',
        'volatility_spike': 'Volatility Spike: Returns multiplied by 2x',
        'correlation_spike': 'Correlation Spike: All stocks move together (full correlation)'
    }
    
    return descriptions.get(scenario_type, 'Unknown scenario')
