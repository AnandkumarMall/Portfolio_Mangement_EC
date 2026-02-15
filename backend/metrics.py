"""
Performance metrics calculation module.
"""

import pandas as pd
import numpy as np
from typing import Dict


def compute_total_return(equity_curve: pd.Series) -> float:
    """
    Compute total return from equity curve.
    
    Args:
        equity_curve: Portfolio value over time
        
    Returns:
        Total return as decimal (e.g., 0.5 = 50%)
    """
    if len(equity_curve) == 0:
        return 0.0
    
    initial_value = equity_curve.iloc[0]
    final_value = equity_curve.iloc[-1]
    
    return (final_value / initial_value) - 1


def compute_cagr(equity_curve: pd.Series) -> float:
    """
    Compute Compound Annual Growth Rate.
    
    Args:
        equity_curve: Portfolio value over time
        
    Returns:
        CAGR as decimal
    """
    if len(equity_curve) < 2:
        return 0.0
    
    total_return = compute_total_return(equity_curve)
    
    # Compute number of years
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25
    
    if years <= 0:
        return 0.0
    
    # CAGR formula
    cagr = (1 + total_return) ** (1 / years) - 1
    
    return cagr


def compute_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    Compute volatility (standard deviation of returns).
    
    Args:
        returns: Daily returns
        annualize: If True, annualize the volatility
        
    Returns:
        Volatility as decimal
    """
    if len(returns) == 0:
        return 0.0
    
    vol = returns.std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Compute Sharpe Ratio.
    
    Args:
        returns: Daily returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Annualized excess return
    excess_return = returns.mean() * 252 - risk_free_rate
    
    # Annualized volatility
    vol = compute_volatility(returns, annualize=True)
    
    if vol == 0:
        return 0.0
    
    sharpe = excess_return / vol
    
    return sharpe


def compute_sortino_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.0,
    target_return: float = 0.0
) -> float:
    """
    Compute Sortino Ratio (uses downside deviation).
    
    Args:
        returns: Daily returns
        risk_free_rate: Annual risk-free rate
        target_return: Target return threshold
        
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Annualized excess return
    excess_return = returns.mean() * 252 - risk_free_rate
    
    # Downside deviation (only consider returns below target)
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) == 0:
        return 0.0
    
    downside_deviation = downside_returns.std() * np.sqrt(252)
    
    if downside_deviation == 0:
        return 0.0
    
    sortino = excess_return / downside_deviation
    
    return sortino


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute maximum drawdown.
    
    Args:
        equity_curve: Portfolio value over time
        
    Returns:
        Maximum drawdown as negative decimal (e.g., -0.2 = -20%)
    """
    if len(equity_curve) == 0:
        return 0.0
    
    from feature_engineering import compute_portfolio_drawdown
    
    drawdown = compute_portfolio_drawdown(equity_curve)
    max_dd = drawdown.min()
    
    return max_dd


def compute_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """
    Compute Calmar Ratio (CAGR / |Max Drawdown|).
    
    Args:
        cagr: Compound annual growth rate
        max_drawdown: Maximum drawdown (negative value)
        
    Returns:
        Calmar ratio
    """
    if max_drawdown >= 0:
        return 0.0
    
    calmar = cagr / abs(max_drawdown)
    
    return calmar


def compute_time_in_cash(exposure_timeline: pd.Series, threshold: float = 0.2) -> float:
    """
    Compute percentage of time with exposure below threshold.
    
    Args:
        exposure_timeline: Series of portfolio exposure over time
        threshold: Exposure threshold for "cash" definition
        
    Returns:
        Percentage of time in cash (0-100)
    """
    if len(exposure_timeline) == 0:
        return 0.0
    
    in_cash = exposure_timeline < threshold
    pct_in_cash = in_cash.sum() / len(exposure_timeline) * 100
    
    return pct_in_cash


def compute_win_rate(returns: pd.Series) -> float:
    """
    Compute percentage of positive return days.
    
    Args:
        returns: Daily returns
        
    Returns:
        Win rate as percentage (0-100)
    """
    if len(returns) == 0:
        return 0.0
    
    positive_days = (returns > 0).sum()
    win_rate = positive_days / len(returns) * 100
    
    return win_rate


def compute_all_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    exposure_timeline: pd.Series,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Compute all performance metrics.
    
    Args:
        equity_curve: Portfolio value over time
        returns: Daily returns
        exposure_timeline: Portfolio exposure over time
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {
        'Total Return': compute_total_return(equity_curve),
        'CAGR': compute_cagr(equity_curve),
        'Volatility': compute_volatility(returns, annualize=True),
        'Sharpe Ratio': compute_sharpe_ratio(returns, risk_free_rate),
        'Sortino Ratio': compute_sortino_ratio(returns, risk_free_rate),
        'Max Drawdown': compute_max_drawdown(equity_curve),
        'Calmar Ratio': compute_calmar_ratio(
            compute_cagr(equity_curve),
            compute_max_drawdown(equity_curve)
        ),
        'Time in Cash (%)': compute_time_in_cash(exposure_timeline),
        'Win Rate (%)': compute_win_rate(returns),
    }
    
    return metrics


def format_metrics(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Format metrics as a nice DataFrame for display.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Formatted DataFrame
    """
    df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    
    # Format percentages
    pct_metrics = ['Total Return', 'CAGR', 'Volatility', 'Max Drawdown', 
                   'Time in Cash (%)', 'Win Rate (%)']
    
    for metric in pct_metrics:
        if metric in df.index:
            if 'Time in Cash' in metric or 'Win Rate' in metric:
                df.loc[metric, 'Formatted'] = f"{df.loc[metric, 'Value']:.2f}%"
            else:
                df.loc[metric, 'Formatted'] = f"{df.loc[metric, 'Value']*100:.2f}%"
    
    # Format ratios
    ratio_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
    for metric in ratio_metrics:
        if metric in df.index:
            df.loc[metric, 'Formatted'] = f"{df.loc[metric, 'Value']:.3f}"
    
    return df
