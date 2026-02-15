"""
Data loading module for fetching and caching stock price data.
Uses yfinance for data retrieval and local CSV caching.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import List
import os
from datetime import datetime

from config import TICKERS, DATA_RAW_PATH


def download_data(
    tickers: List[str], 
    start_date: str, 
    end_date: str,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Download adjusted close prices for given tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        force_download: If True, download even if cached data exists
        
    Returns:
        DataFrame with dates as index and tickers as columns
    """
    # Create cache directory if it doesn't exist
    Path(DATA_RAW_PATH).mkdir(parents=True, exist_ok=True)
    
    # Define cache file name
    cache_file = f"{DATA_RAW_PATH}/prices_{start_date}_{end_date}.csv"
    
    # Check if cached data exists
    if not force_download and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    # Download data using yfinance
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True  # Use adjusted close
    )
    
    # Extract 'Close' prices (already adjusted)
    if len(tickers) == 1:
        prices = data[['Close']].copy()
        prices.columns = tickers
    else:
        prices = data['Close'].copy()
    
    # Save to cache
    prices.to_csv(cache_file)
    print(f"Data saved to {cache_file}")
    
    return prices


def load_cached_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load cached price data if available.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with prices or None if not cached
    """
    cache_file = f"{DATA_RAW_PATH}/prices_{start_date}_{end_date}.csv"
    
    if os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        print("No cached data found")
        return None


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns from price data.
    
    Args:
        prices: DataFrame of prices
        
    Returns:
        DataFrame of daily returns
    """
    returns = prices.pct_change()
    return returns


def handle_missing_data(data: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        data: DataFrame with potential missing values
        method: Method to handle NaNs ('ffill', 'bfill', 'drop')
        
    Returns:
        DataFrame with handled missing values
    """
    if method == 'ffill':
        # Forward fill missing values
        data_cleaned = data.fillna(method='ffill')
    elif method == 'bfill':
        # Backward fill missing values
        data_cleaned = data.fillna(method='bfill')
    elif method == 'drop':
        # Drop rows with any NaN
        data_cleaned = data.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Check for remaining NaNs
    if data_cleaned.isna().sum().sum() > 0:
        print(f"Warning: {data_cleaned.isna().sum().sum()} NaN values remain after {method}")
        # Drop remaining NaNs
        data_cleaned = data_cleaned.dropna()
    
    return data_cleaned


def validate_data_quality(data: pd.DataFrame, max_missing_pct: float = 0.05) -> bool:
    """
    Validate data quality by checking for excessive missing values.
    
    Args:
        data: DataFrame to validate
        max_missing_pct: Maximum allowed percentage of missing values
        
    Returns:
        True if data quality is acceptable
    """
    total_values = data.shape[0] * data.shape[1]
    missing_values = data.isna().sum().sum()
    missing_pct = missing_values / total_values
    
    if missing_pct > max_missing_pct:
        print(f"Warning: {missing_pct*100:.2f}% missing values exceeds threshold of {max_missing_pct*100:.2f}%")
        return False
    
    print(f"Data quality check passed: {missing_pct*100:.2f}% missing values")
    return True


def get_data(start_year: int, end_year: int, force_download: bool = False) -> tuple:
    """
    Main function to get prices and returns data.
    
    Args:
        start_year: Start year
        end_year: End year
        force_download: Force re-download of data
        
    Returns:
        Tuple of (prices, returns) DataFrames
    """
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    # Download or load cached data
    prices = download_data(TICKERS, start_date, end_date, force_download)
    
    # Validate data quality
    validate_data_quality(prices)
    
    # Handle missing data
    prices = handle_missing_data(prices, method='ffill')
    
    # Compute returns
    returns = compute_returns(prices)
    
    # Drop first row (NaN from pct_change)
    returns = returns.iloc[1:]
    
    print(f"Loaded data: {len(prices)} days, {len(prices.columns)} tickers")
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")
    
    return prices, returns
