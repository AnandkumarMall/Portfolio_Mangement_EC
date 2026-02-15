"""
Configuration constants for the Adaptive Portfolio Engine.
All parameters are centralized here for easy modification.
"""

from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

# ========== Asset Universe ==========
# 20 Diversified US Stocks (S&P 500)
TICKERS: List[str] = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "NVDA",
    # Finance
    "JPM", "BAC", "GS", "V",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV",
    # Consumer
    "WMT", "HD", "MCD", "NKE",
    # Energy & Industrials
    "XOM", "CVX", "CAT", "BA"
]

# Currency
CURRENCY_SYMBOL = "$"
CURRENCY_NAME = "USD"

# ========== Date Range ==========
DEFAULT_START_YEAR: int = int(os.getenv("DEFAULT_START_YEAR", "2010"))
DEFAULT_END_YEAR: int = int(os.getenv("DEFAULT_END_YEAR", "2024"))

# ========== Transaction Costs ==========
TRANSACTION_COST: float = float(os.getenv("TRANSACTION_COST", "0.001"))  # 0.1%

# ========== Rolling Window Sizes ==========
WINDOW_SHORT: int = 21   # ~1 month
WINDOW_MEDIUM: int = 63  # ~3 months
WINDOW_LONG: int = 252   # ~1 year

# ========== Moving Average Windows ==========
MA_SHORT: int = 50
MA_LONG: int = 200

# ========== Rebalancing ==========
REBALANCE_FREQUENCY: str = "monthly"  # Monthly rebalancing

# ========== Regime Definitions ==========
REGIME_THRESHOLDS: Dict[str, float] = {
    "crash_drawdown": 0.15,      # 15% drawdown triggers crash regime
    "volatile_vol": 0.02,         # 2% daily vol triggers volatile regime
    "bull_vol": 0.015,            # Below 1.5% daily vol for bull
}

# ========== Regime-Based Exposure ==========
REGIME_EXPOSURE: Dict[str, float] = {
    "BULL": 1.0,      # 100% stocks
    "VOLATILE": 0.6,  # 60% stocks
    "CRASH": 0.2,     # 20% stocks
}

# ========== Risk Engine Parameters ==========
RISK_PARAMS: Dict[str, float] = {
    "target_volatility": 0.20,      # 20% annualized target
    "volatility_adjustment": 0.15,  # Scale down to 15% if exceeded
    "drawdown_threshold_1": 0.10,   # 10% DD → reduce by 50%
    "drawdown_threshold_2": 0.20,   # 20% DD → move to cash
    "max_position_weight": 0.10,    # 10% max per stock
    "stop_loss_threshold": -0.05,   # -5% daily return triggers stop-loss
}

# ========== Walk-Forward Parameters ==========
TRAIN_WINDOW_YEARS: int = 3
TEST_WINDOW_YEARS: int = 1

# ========== Data Paths ==========
DATA_RAW_PATH: str = "data/raw"
DATA_PROCESSED_PATH: str = "data/processed"

# ========== Risk-Free Rate ==========
RISK_FREE_RATE: float = 0.0  # Assume 0 for simplicity
