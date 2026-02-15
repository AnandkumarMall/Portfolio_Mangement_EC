"""
Walk-forward backtesting engine with daily progression and monthly rebalancing.
This is the core simulation module that ties everything together.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

from config import (
    TRAIN_WINDOW_YEARS, TEST_WINDOW_YEARS, TRANSACTION_COST,
    TICKERS, REGIME_EXPOSURE
)
from feature_engineering import compute_all_features, compute_portfolio_drawdown
from regime_detection import detect_regime_single_date
from allocation_engine import compute_allocation
from risk_engine import apply_risk_controls
from explainability import create_regime_change_log


def is_month_start(date, dates_index):
    """Check if date is first business day of month."""
    idx = dates_index.get_loc(date)
    if idx == 0:
        return True
    
    prev_date = dates_index[idx - 1]
    return date.month != prev_date.month


def run_walkforward_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    start_year: int,
    end_year: int,
    with_risk_engine: bool = True,
    allocation_method: str = 'inverse_vol'
) -> Dict:
    """
    Run walk-forward backtest with daily progression.
    
    Args:
        prices: Price DataFrame
        returns: Returns DataFrame
        start_year: Start year for backtest
        end_year: End year for backtest
        with_risk_engine: Whether to apply risk controls
        allocation_method: 'inverse_vol' or 'equal_weight'
        
    Returns:
        Dictionary containing:
        - equity_curve: Daily portfolio values
        - drawdown_curve: Daily drawdowns
        - exposure_timeline: Daily exposure percentages
        - regime_timeline: Daily regime labels
        - risk_logs: List of risk events
        - metrics: Performance metrics dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running Walk-Forward Backtest")
    print(f"Period: {start_year}-{end_year}")
    print(f"Risk Engine: {'ON' if with_risk_engine else 'OFF'}")
    print(f"Allocation Method: {allocation_method}")
    print(f"{'='*60}\n")
    
    # Filter data to backtest period
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    prices_bt = prices.loc[start_date:end_date]
    returns_bt = returns.loc[start_date:end_date]
    
    # Compute features (these are lagged by 1 day)
    features = compute_all_features(prices_bt, returns_bt)
    
    # Initialize portfolio state
    portfolio_value = 100000  # Start with $100k
    portfolio_values = []
    dates = []
    exposures = []
    regimes_list = []
    risk_logs = []
    
    current_weights = pd.Series(0.0, index=TICKERS)
    previous_regime = None
    
    # Iterate through each day
    for i, date in enumerate(returns_bt.index):
        dates.append(date)
        
        # Detect regime for this date
        # Use features from this date (which are already lagged)
        vol_21d = features['volatility_21d'].loc[date].mean()
        mom_252d = features['momentum_252d'].loc[date].mean()
        
        # Compute drawdown up to this point
        if len(portfolio_values) > 0:
            equity_series = pd.Series(portfolio_values, index=dates[:-1])
            current_dd = compute_portfolio_drawdown(equity_series).iloc[-1]
        else:
            current_dd = 0.0
        
        regime = detect_regime_single_date(vol_21d, mom_252d, current_dd)
        regimes_list.append(regime)
        
        # Log regime changes
        if previous_regime is not None and regime != previous_regime:
            log = create_regime_change_log(date, previous_regime, regime)
            risk_logs.append(log)
        previous_regime = regime
        
        # Check if we should rebalance (month start)
        should_rebalance = is_month_start(date, returns_bt.index)
        
        if should_rebalance:
            # Compute new allocation
            stock_vols = features['volatility_21d'].loc[date]
            
            # Get base allocation from allocation engine
            new_weights = compute_allocation(
                regime=regime,
                volatilities=stock_vols,
                method=allocation_method
            )
            
            # Apply risk controls if enabled
            if with_risk_engine:
                portfolio_state = {
                    'equity_curve': portfolio_values.copy(),
                    'dates': dates[:-1].copy()
                }
                
                new_weights, risk_events = apply_risk_controls(
                    weights=new_weights,
                    portfolio_state=portfolio_state,
                    features=features,
                    returns=returns_bt,
                    date=date
                )
                
                # Add risk events to logs
                risk_logs.extend(risk_events)
            
            # Compute transaction costs
            turnover = (new_weights - current_weights).abs().sum()
            transaction_cost = portfolio_value * turnover * TRANSACTION_COST
            portfolio_value -= transaction_cost
            
            # Update weights
            current_weights = new_weights
        
        # Compute daily portfolio return
        daily_returns = returns_bt.loc[date]
        portfolio_return = (current_weights * daily_returns).sum()
        
        # Update portfolio value
        portfolio_value = portfolio_value * (1 + portfolio_return)
        portfolio_values.append(portfolio_value)
        
        # Track exposure
        current_exposure = current_weights.sum()
        exposures.append(current_exposure)
        
        # Progress indicator
        if (i + 1) % 252 == 0:
            years_complete = (i + 1) // 252
            print(f"Processed {years_complete} year(s), Portfolio Value: ${portfolio_value:,.2f}")
    
    # Create equity curve series
    equity_curve = pd.Series(portfolio_values, index=dates)
    
    # Compute drawdown curve
    drawdown_curve = compute_portfolio_drawdown(equity_curve)
    
    # Create exposure timeline
    exposure_timeline = pd.Series(exposures, index=dates)
    
    # Create regime timeline
    regime_timeline = pd.Series(regimes_list, index=dates)
    
    # Compute portfolio returns for metrics
    portfolio_returns = equity_curve.pct_change().dropna()
    
    print(f"\nBacktest Complete!")
    print(f"Final Portfolio Value: ${portfolio_value:,.2f}")
    print(f"Total Return: {(portfolio_value/100000 - 1)*100:.2f}%")
    
    return {
        'equity_curve': equity_curve,
        'drawdown_curve': drawdown_curve,
        'exposure_timeline': exposure_timeline,
        'regime_timeline': regime_timeline,
        'risk_logs': risk_logs,
        'portfolio_returns': portfolio_returns
    }


def run_simple_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    start_year: int,
    end_year: int,
    with_risk_engine: bool = True
) -> Dict:
    """
    Simplified wrapper for backtesting (no walk-forward, full period).
    
    Args:
        prices: Price DataFrame
        returns: Returns DataFrame
        start_year: Start year
        end_year: End year
        with_risk_engine: Whether to use risk engine
        
    Returns:
        Backtest results dictionary
    """
    return run_walkforward_backtest(
        prices=prices,
        returns=returns,
        start_year=start_year,
        end_year=end_year,
        with_risk_engine=with_risk_engine,
        allocation_method='inverse_vol'
    )
