"""
Explainability module for logging risk events and regime changes.
"""

import pandas as pd
from typing import Dict, List
from datetime import datetime


def log_event(
    date: pd.Timestamp,
    event_type: str,
    details: str,
    logs_list: List[Dict],
    **kwargs
) -> List[Dict]:
    """
    Log a risk or regime event.
    
    Args:
        date: Date of the event
        event_type: Type of event (REGIME_CHANGE, VOL_BREACH, etc.)
        details: Human-readable description
        logs_list: Existing list of logs to append to
        **kwargs: Additional event-specific data
        
    Returns:
        Updated logs list
    """
    event = {
        'date': str(date.date()) if isinstance(date, pd.Timestamp) else str(date),
        'event_type': event_type,
        'details': details,
        **kwargs
    }
    
    logs_list.append(event)
    
    return logs_list


def create_regime_change_log(
    date: pd.Timestamp,
    from_regime: str,
    to_regime: str
) -> Dict:
    """
    Create a log entry for regime change.
    
    Args:
        date: Date of regime change
        from_regime: Previous regime
        to_regime: New regime
        
    Returns:
        Log dictionary
    """
    return {
        'date': str(date.date()) if isinstance(date, pd.Timestamp) else str(date),
        'event_type': 'REGIME_CHANGE',
        'details': f'Regime changed from {from_regime} to {to_regime}',
        'from_regime': from_regime,
        'to_regime': to_regime
    }


def merge_logs(logs_list: List[List[Dict]]) -> List[Dict]:
    """
    Merge multiple log lists into one sorted list.
    
    Args:
        logs_list: List of log lists
        
    Returns:
        Merged and sorted list of logs
    """
    merged = []
    for logs in logs_list:
        merged.extend(logs)
    
    # Sort by date
    merged.sort(key=lambda x: x['date'])
    
    return merged


def logs_to_dataframe(logs: List[Dict]) -> pd.DataFrame:
    """
    Convert logs to DataFrame for easy viewing.
    
    Args:
        logs: List of log dictionaries
        
    Returns:
        DataFrame of logs
    """
    if not logs:
        return pd.DataFrame(columns=['date', 'event_type', 'details'])
    
    df = pd.DataFrame(logs)
    
    # Ensure date column exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    return df


def filter_logs_by_type(logs: List[Dict], event_types: List[str]) -> List[Dict]:
    """
    Filter logs by event type.
    
    Args:
        logs: List of log dictionaries
        event_types: List of event types to include
        
    Returns:
        Filtered list of logs
    """
    filtered = [log for log in logs if log.get('event_type') in event_types]
    return filtered


def filter_logs_by_date_range(
    logs: List[Dict],
    start_date: str,
    end_date: str
) -> List[Dict]:
    """
    Filter logs by date range.
    
    Args:
        logs: List of log dictionaries
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Filtered list of logs
    """
    filtered = [
        log for log in logs
        if start_date <= log['date'] <= end_date
    ]
    return filtered


def get_event_summary(logs: List[Dict]) -> pd.DataFrame:
    """
    Get summary statistics of events.
    
    Args:
        logs: List of log dictionaries
        
    Returns:
        DataFrame with event counts
    """
    if not logs:
        return pd.DataFrame(columns=['Event Type', 'Count'])
    
    df = pd.DataFrame(logs)
    
    if 'event_type' not in df.columns:
        return pd.DataFrame(columns=['Event Type', 'Count'])
    
    summary = df['event_type'].value_counts().reset_index()
    summary.columns = ['Event Type', 'Count']
    
    return summary
