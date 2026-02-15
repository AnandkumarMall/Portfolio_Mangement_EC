# Portfolio Mangement

A production-grade, rule-based portfolio allocation and risk management system with walk-forward backtesting, stress testing, and explainability.

## 🎯 What This IS

* **Adaptive Portfolio Allocation Engine**: Dynamically adjusts stock exposure based on market regime (BULL/VOLATILE/CRASH)
* **Walk-Forward Backtesting System**: Tests strategy across multiple market conditions with proper out-of-sample validation
* **Stress Testing Framework**: Simulates extreme market conditions (shocks, volatility spikes, correlation breaks)
* **Risk-Controlled Capital Allocation**: Multi-layered risk management (volatility targeting, drawdown protection, position limits, stop-loss)

## 🚫 What This is NOT

* Not a stock price predictor
* Not a machine learning trading bot
* Not a buy/sell signal generator
* No ML, no XGBoost, no neural networks

## 🏗️ Architecture

```
adaptive_portfolio_engine/
│
├── data/
│   ├── raw/              # Cached yfinance downloads
│   └── processed/        # Computed features
│
├── backend/              # FastAPI backend
│   ├── main.py          # API endpoints
│   ├── config.py        # Configuration
│   ├── data_loader.py   # yfinance integration
│   ├── feature_engineering.py
│   ├── regime_detection.py
│   ├── allocation_engine.py
│   ├── risk_engine.py
│   ├── backtester.py
│   ├── stress_testing.py
│   ├── metrics.py
│   └── explainability.py
│
├── frontend/             # Streamlit dashboard
│   └── app.py
│
├── requirements.txt
├── README.md
└── .env
```

## 📦 Installation

### Prerequisites

You already have a `.venv` virtual environment. Activate it:

```bash
# Windows
.venv\Scripts\activate

# After activation, install dependencies
pip install -r adaptive_portfolio_engine/requirements.txt
```

## 🚀 Running the Application

### 1. Start the Backend (FastAPI)

Open a terminal in the project directory:

```bash
cd adaptive_portfolio_engine
python backend/main.py
```

Or using uvicorn directly:

```bash
cd adaptive_portfolio_engine
uvicorn backend.main:app --reload
```

The API will be available at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### 2. Start the Frontend (Streamlit)

Open a **separate terminal** in the project directory:

```bash
cd adaptive_portfolio_engine
streamlit run frontend/app.py
```

The dashboard will open automatically in your browser at: `http://localhost:8501`

## 🎨 Features

### 1. Regime Detection (Rule-Based)

Classifies market conditions into three regimes:

| Regime | Definition | Stock Exposure |
|--------|-----------|----------------|
| **BULL** | Low volatility + positive momentum | 100% |
| **VOLATILE** | High volatility | 60% |
| **CRASH** | Drawdown > 15% | 20% |

### 2. Allocation Engine

- **Inverse Volatility Weighting**: Allocate more to less volatile stocks
- **Regime-Based Exposure**: Scale total exposure based on regime
- **Position Limits**: Max 10% per stock

### 3. Risk Management Engine

Multi-layered protection:

1. **Volatility Targeting**: Scale down if portfolio vol > 20% (target: 15%)
2. **Drawdown Protection**:
   - Drawdown > 10% → Reduce exposure by 50%
   - Drawdown > 20% → Move to 100% cash
3. **Position Caps**: Max 10% per stock
4. **Stop-Loss**: Exit positions with daily loss > 5%

### 4. Walk-Forward Backtesting

- **Train Window**: 3 years
- **Test Window**: 1 year
- **Daily Progression**: Updates portfolio value every day
- **Monthly Rebalancing**: Trades on 1st business day of month
- **Transaction Costs**: 0.1% per trade
- **No Lookahead Bias**: All features use `.shift(1)`

### 5. Stress Testing

Three scenarios:

1. **Market Shock**: Subtract 5% from all daily returns
2. **Volatility Spike**: Multiply returns by 2x
3. **Correlation Spike**: Force all stocks to move together

### 6. Performance Metrics

- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Calmar Ratio
- Volatility
- Time in Cash
- Win Rate

## 📊 Using the Dashboard

### Run Backtest

1. Set date range in sidebar
2. Toggle "Enable Risk Engine"
3. Click "Run Backtest"
4. View metrics, charts, and risk logs

### Run Stress Test

1. Select stress scenario (Market Shock / Volatility Spike / Correlation Spike)
2. Click "Run Stress Test"
3. Compare stressed vs normal performance

### Compare All Scenarios

Click "Compare All Scenarios" to run 4 backtests:

1. Normal without risk engine
2. Normal with risk engine
3. Stressed without risk engine
4. Stressed with risk engine

## 🔬 Expected Results

### Without Risk Engine

- Higher returns during bull markets
- Severe drawdowns during crashes (>30%)
- High volatility
- Lower Sharpe ratio

### With Risk Engine

- Moderate returns (CAGR: 8-12%)
- Controlled drawdowns (<15%)
- Lower volatility
- Higher Sharpe ratio (>1.0)
- More time in cash during crises

### Under Stress

- Both strategies suffer
- Risk-managed version degrades more gracefully
- Demonstrates value of risk controls

## 🧪 API Endpoints

### GET `/health`

Health check endpoint.

### POST `/run_backtest`

```json
{
  "start_year": 2015,
  "end_year": 2024,
  "with_risk_engine": true
}
```

### POST `/run_stress_test`

```json
{
  "stress_type": "market_shock",
  "with_risk_engine": true,
  "start_year": 2015,
  "end_year": 2024
}
```

### GET `/compare_risk_engine`

Query params: `start_year`, `end_year`

## 📈 Asset Universe

20 diversified US large-cap stocks:

- **Tech**: AAPL, MSFT, GOOGL, NVDA
- **Finance**: JPM, BAC, GS
- **Healthcare**: JNJ, UNH, PFE
- **Consumer**: WMT, HD, MCD, PG
- **Energy**: XOM, CVX
- **Industrials**: BA, CAT, GE
- **Telecom**: T, VZ

## ⚙️ Configuration

Edit `.env` file:

```bash
TRANSACTION_COST=0.001
DEFAULT_START_YEAR=2010
DEFAULT_END_YEAR=2024
BACKEND_URL=http://localhost:8000
```

Modify thresholds in `backend/config.py`:

```python
REGIME_THRESHOLDS = {
    "crash_drawdown": 0.15,
    "volatile_vol": 0.02,
    "bull_vol": 0.015,
}

RISK_PARAMS = {
    "target_volatility": 0.20,
    "drawdown_threshold_1": 0.10,
    "drawdown_threshold_2": 0.20,
    "max_position_weight": 0.10,
    "stop_loss_threshold": -0.05,
}
```

## 🧠 Design Decisions

### Why Rule-Based?

- **Explainability**: Every decision is traceable
- **Robustness**: No overfitting to historical data
- **Practicality**: Easier to maintain in production

### Why Walk-Forward?

- **Realistic**: Prevents lookahead bias
- **Robust**: Tests across different market regimes
- **Honest**: Shows true out-of-sample performance

### Why Daily Progression with Monthly Rebalancing?

- **Realistic**: Reduces transaction costs
- **Practical**: Mimics real portfolio management
- **Flexible**: Can be adjusted in config

## 🐛 Troubleshooting

### Backend Connection Error

Ensure backend is running at `http://localhost:8000`:

```bash
cd adaptive_portfolio_engine
python backend/main.py
```

### Data Download Issues

If yfinance fails, check internet connection or try:

```python
from backend.data_loader import get_data
prices, returns = get_data(2015, 2024, force_download=True)
```

### Module Import Errors

Ensure you're running from the correct directory:

```bash
cd adaptive_portfolio_engine
python -m backend.main
```

Or add to PYTHONPATH:

```bash
set PYTHONPATH=%PYTHONPATH%;C:\Users\Anand Mall\OneDrive\Desktop\Portfoloi_Mangement\adaptive_portfolio_engine
```

## 📝 License

This is a demonstration project for educational purposes.

## 🙏 Acknowledgments

Built with:
- FastAPI
- Streamlit
- yfinance
- Plotly
- pandas/numpy


