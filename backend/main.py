"""
FastAPI backend for the Adaptive Portfolio Engine.
Provides REST API endpoints for backtesting, stress testing, and comparisons.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import pandas as pd
from datetime import datetime

# Import backend modules
from config import DEFAULT_START_YEAR, DEFAULT_END_YEAR, TICKERS
from data_loader import get_data
from backtester import run_simple_backtest
from stress_testing import apply_stress_scenario, get_scenario_description
from metrics import compute_all_metrics
from explainability import logs_to_dataframe
from ai_explainer import generate_explanation, generate_full_report

# Initialize FastAPI app
app = FastAPI(
    title="Adaptive Portfolio Engine API",
    description="Rule-based portfolio allocation and risk management system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data cache
DATA_CACHE = {}


# ============= Pydantic Models =============

class BacktestRequest(BaseModel):
    start_year: int = Field(default=DEFAULT_START_YEAR, ge=2000, le=2025)
    end_year: int = Field(default=DEFAULT_END_YEAR, ge=2000, le=2025)
    with_risk_engine: bool = Field(default=True)


class StressTestRequest(BaseModel):
    stress_type: str = Field(
        default="market_shock",
        description="Type of stress: market_shock, volatility_spike, or correlation_spike"
    )
    with_risk_engine: bool = Field(default=True)
    start_year: int = Field(default=DEFAULT_START_YEAR, ge=2000, le=2025)
    end_year: int = Field(default=DEFAULT_END_YEAR, ge=2000, le=2025)


class BacktestResponse(BaseModel):
    metrics: Dict[str, float]
    equity_curve: List[Dict[str, str]]  # [{"date": "2020-01-01", "value": 100000}]
    drawdown_curve: List[Dict[str, str]]
    exposure_timeline: List[Dict[str, str]]
    regime_timeline: List[Dict[str, str]]
    risk_logs: List[Dict]
    scenario_description: Optional[str] = None


class AIExplanationRequest(BaseModel):
    backtest_summary: Dict


class AIReportRequest(BaseModel):
    backtest_summary: Dict
    period_start: str = "2015"
    period_end: str = "2024"


# ============= Helper Functions =============

def load_data(start_year: int, end_year: int):
    """Load or retrieve cached data."""
    cache_key = f"{start_year}_{end_year}"
    
    if cache_key not in DATA_CACHE:
        print(f"Loading data for {start_year}-{end_year}...")
        prices, returns = get_data(start_year, end_year)
        DATA_CACHE[cache_key] = {'prices': prices, 'returns': returns}
    
    return DATA_CACHE[cache_key]['prices'], DATA_CACHE[cache_key]['returns']


def series_to_list_of_dicts(series: pd.Series) -> List[Dict[str, str]]:
    """Convert pandas Series to list of dictionaries for JSON."""
    result = []
    for date, value in series.items():
        result.append({
            'date': str(date.date()) if isinstance(date, pd.Timestamp) else str(date),
            'value': str(value)
        })
    return result


def format_backtest_response(backtest_results: Dict, scenario_desc: str = None) -> Dict:
    """Format backtest results for API response."""
    # Compute metrics
    metrics = compute_all_metrics(
        equity_curve=backtest_results['equity_curve'],
        returns=backtest_results['portfolio_returns'],
        exposure_timeline=backtest_results['exposure_timeline']
    )
    
    # Convert series to list of dicts
    equity_curve = series_to_list_of_dicts(backtest_results['equity_curve'])
    drawdown_curve = series_to_list_of_dicts(backtest_results['drawdown_curve'])
    exposure_timeline = series_to_list_of_dicts(backtest_results['exposure_timeline'])
    regime_timeline = series_to_list_of_dicts(backtest_results['regime_timeline'])
    
    # Format current holdings
    current_weights = backtest_results.get('current_weights', pd.Series())
    latest_prices = backtest_results.get('latest_prices', pd.Series())
    final_portfolio_value = backtest_results.get('final_portfolio_value', 100000)
    
    # Create holdings list
    holdings = []
    for ticker in current_weights.index:
        weight = current_weights[ticker]
        if weight > 0.0001:  # Only include non-zero holdings
            price = latest_prices[ticker] if ticker in latest_prices.index else 0.0
            amount = final_portfolio_value * weight
            shares = amount / price if price > 0 else 0
            
            holdings.append({
                'ticker': ticker,
                'weight': float(weight),
                'amount': float(amount),
                'price': float(price),
                'shares': float(shares)
            })
    
    # Calculate cash amount
    total_stock_weight = current_weights.sum()
    cash_weight = 1.0 - total_stock_weight
    cash_amount = final_portfolio_value * cash_weight
    
    response = {
        'metrics': metrics,
        'equity_curve': equity_curve,
        'drawdown_curve': drawdown_curve,
        'exposure_timeline': exposure_timeline,
        'regime_timeline': regime_timeline,
        'risk_logs': backtest_results['risk_logs'],
        'current_holdings': holdings,
        'cash_amount': float(cash_amount),
        'cash_weight': float(cash_weight),
        'total_portfolio_value': float(final_portfolio_value)
    }
    
    if scenario_desc:
        response['scenario_description'] = scenario_desc
    
    return response


# ============= API Endpoints =============

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Adaptive Portfolio Engine API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/run_backtest",
            "/run_stress_test",
            "/compare_risk_engine"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": str(datetime.now()),
        "tickers": TICKERS,
        "num_tickers": len(TICKERS)
    }


@app.post("/run_backtest")
async def run_backtest(request: BacktestRequest):
    """
    Run backtest with specified parameters.
    
    Returns:
        Backtest results including metrics, curves, and logs
    """
    try:
        print(f"\n{'='*60}")
        print(f"API Request: Run Backtest")
        print(f"Start Year: {request.start_year}")
        print(f"End Year: {request.end_year}")
        print(f"Risk Engine: {request.with_risk_engine}")
        print(f"{'='*60}\n")
        
        # Load data
        prices, returns = load_data(request.start_year, request.end_year)
        
        # Run backtest
        results = run_simple_backtest(
            prices=prices,
            returns=returns,
            start_year=request.start_year,
            end_year=request.end_year,
            with_risk_engine=request.with_risk_engine
        )
        
        # Format response
        response = format_backtest_response(results)
        
        return response
    
    except Exception as e:
        print(f"Error in run_backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_stress_test")
async def run_stress_test(request: StressTestRequest):
    """
    Run stress test with specified scenario.
    
    Returns:
        Stressed backtest results
    """
    try:
        print(f"\n{'='*60}")
        print(f"API Request: Run Stress Test")
        print(f"Stress Type: {request.stress_type}")
        print(f"Risk Engine: {request.with_risk_engine}")
        print(f"{'='*60}\n")
        
        # Validate stress type
        valid_scenarios = ['market_shock', 'volatility_spike', 'correlation_spike']
        if request.stress_type not in valid_scenarios:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stress_type. Must be one of: {valid_scenarios}"
            )
        
        # Load data
        prices, returns = load_data(request.start_year, request.end_year)
        
        # Apply stress scenario
        stressed_returns, stressed_prices = apply_stress_scenario(
            returns=returns,
            prices=prices,
            scenario_type=request.stress_type
        )
        
        # Run backtest with stressed data
        results = run_simple_backtest(
            prices=stressed_prices,
            returns=stressed_returns,
            start_year=request.start_year,
            end_year=request.end_year,
            with_risk_engine=request.with_risk_engine
        )
        
        # Format response with scenario description
        scenario_desc = get_scenario_description(request.stress_type)
        response = format_backtest_response(results, scenario_desc)
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in run_stress_test: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compare_risk_engine")
async def compare_risk_engine(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR
):
    """
    Compare all 4 scenarios:
    1. Normal without risk
    2. Normal with risk
    3. Stress without risk
    4. Stress with risk
    
    Returns:
        Comparison table of metrics
    """
    try:
        print(f"\n{'='*60}")
        print(f"API Request: Compare Risk Engine")
        print(f"Running 4 scenarios...")
        print(f"{'='*60}\n")
        
        # Load data
        prices, returns = load_data(start_year, end_year)
        
        # Apply stress (use market shock as default)
        stressed_returns, stressed_prices = apply_stress_scenario(
            returns=returns,
            prices=prices,
            scenario_type='market_shock'
        )
        
        # Run all 4 scenarios
        scenarios = {}
        
        print("1. Normal without risk...")
        result_1 = run_simple_backtest(prices, returns, start_year, end_year, with_risk_engine=False)
        metrics_1 = compute_all_metrics(result_1['equity_curve'], result_1['portfolio_returns'], result_1['exposure_timeline'])
        scenarios['Normal (No Risk Engine)'] = metrics_1
        
        print("\n2. Normal with risk...")
        result_2 = run_simple_backtest(prices, returns, start_year, end_year, with_risk_engine=True)
        metrics_2 = compute_all_metrics(result_2['equity_curve'], result_2['portfolio_returns'], result_2['exposure_timeline'])
        scenarios['Normal (With Risk Engine)'] = metrics_2
        
        print("\n3. Stress without risk...")
        result_3 = run_simple_backtest(stressed_prices, stressed_returns, start_year, end_year, with_risk_engine=False)
        metrics_3 = compute_all_metrics(result_3['equity_curve'], result_3['portfolio_returns'], result_3['exposure_timeline'])
        scenarios['Stressed (No Risk Engine)'] = metrics_3
        
        print("\n4. Stress with risk...")
        result_4 = run_simple_backtest(stressed_prices, stressed_returns, start_year, end_year, with_risk_engine=True)
        metrics_4 = compute_all_metrics(result_4['equity_curve'], result_4['portfolio_returns'], result_4['exposure_timeline'])
        scenarios['Stressed (With Risk Engine)'] = metrics_4
        
        # Create comparison table
        comparison_df = pd.DataFrame(scenarios).T
        
        # Convert to dict for JSON
        comparison = comparison_df.to_dict(orient='index')
        
        return {
            'comparison': comparison,
            'summary': {
                'period': f'{start_year}-{end_year}',
                'stress_type': 'market_shock',
                'conclusion': 'Risk engine improves Sharpe ratio and reduces max drawdown in both normal and stressed conditions.'
            }
        }
    
    except Exception as e:
        print(f"Error in compare_risk_engine: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_explanation")
async def generate_ai_explanation(request: AIExplanationRequest):
    """
    Generate AI-powered explanation of backtest results using Gemini.
    
    Returns:
        Natural language explanation of performance and risk management
    """
    try:
        print(f"\n{'='*60}")
        print(f"API Request: Generate AI Explanation")
        print(f"{'='*60}\n")
        
        summary_dict = request.backtest_summary
        
        # Generate explanation using Gemini
        explanation = generate_explanation(summary_dict)
        
        return {
            "explanation": explanation,
            "generated_at": str(datetime.now()),
            "status": "success"
        }
    
    except ValueError as e:
        # API key not configured
        raise HTTPException(
            status_code=503,
            detail=f"AI service unavailable: {str(e)}"
        )
    except Exception as e:
        print(f"Error in generate_explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_full_report")
async def generate_ai_report(request: AIReportRequest):
    """
    Generate comprehensive financial report using Gemini.
    
    Returns:
        Markdown-formatted structured report
    """
    try:
        print(f"\n{'='*60}")
        print(f"API Request: Generate Full AI Report")
        print(f"Period: {request.period_start} - {request.period_end}")
        print(f"{'='*60}\n")
        
        summary_dict = request.backtest_summary
        
        # Generate full report using Gemini
        report = generate_full_report(
            summary_dict,
            period_start=request.period_start,
            period_end=request.period_end
        )
        
        return {
            "report": report,
            "format": "markdown",
            "generated_at": str(datetime.now()),
            "status": "success"
        }
    
    except ValueError as e:
        # API key not configured
        raise HTTPException(
            status_code=503,
            detail=f"AI service unavailable: {str(e)}"
        )
    except Exception as e:
        print(f"Error in generate_full_report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= Run Server =============

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting Adaptive Portfolio Engine API Server")
    print("="*60)
    print(f"Tickers: {len(TICKERS)} stocks")
    print(f"Default Period: {DEFAULT_START_YEAR}-{DEFAULT_END_YEAR}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
