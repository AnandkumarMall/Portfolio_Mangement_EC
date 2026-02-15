"""
AI Report Generator using Gemini 2.5 Flash and LangChain LCEL.
Generates comprehensive, structured financial reports in Markdown format.
"""

import os
from typing import Dict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from datetime import datetime

load_dotenv()


def get_gemini_model():
    """Initialize Gemini 2.5 Flash model for report generation."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY not configured in .env file")
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2  # Very low temperature for consistent, factual reports
    )


def generate_full_report(summary_dict: Dict, period_start: str = "2015", period_end: str = "2024") -> str:
    """
    Generate a comprehensive, structured financial report in Markdown format.
    
    Args:
        summary_dict: Summarized backtest metrics and events
        period_start: Start year of backtest
        period_end: End year of backtest
        
    Returns:
        Markdown-formatted comprehensive report
    """
    try:
        model = get_gemini_model()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior financial analyst preparing a comprehensive portfolio analysis report.

Generate a professional, structured Markdown report with these exact sections:

# 1. Executive Summary
Brief overview of performance and objectives (2-3 sentences).

# 2. Strategy Description  
Explain the rule-based adaptive allocation approach (1 paragraph).

# 3. Performance Metrics
Summarize key metrics with clear interpretation.

# 4. Market Regime Behavior
Explain how the portfolio adapted across Bull, Volatile, and Crash periods.

# 5. Risk Management Evaluation
Analyze risk engine effectiveness - protection vs. return tradeoff.

# 6. Capital Growth & Stability Analysis
Discuss CAGR vs. drawdown balance and long-term compounding.

# 7. System Strengths
List 3-4 key advantages with brief explanations.

# 8. System Limitations
Mention 2-3 realistic constraints or challenges.

# 9. Final Assessment
Professional evaluation of robustness and suitability (2-3 sentences).

CRITICAL RULES:
- Use ONLY the provided data - do not invent numbers
- Be professional and structured
- Use markdown formatting (headers, lists, bold for emphasis)
- Be factual and balanced
- Mention both strengths AND weaknesses
- No hallucinations or speculation"""),
            ("human", """Generate a comprehensive report for this portfolio backtest:

**Period**: {period_start} - {period_end}

**Performance Metrics**:
- Initial Capital: ₹{initial_capital:,.0f}
- Final Value: ₹{final_portfolio_value:,.0f}
- Total Return: {total_return_pct:.2f}%
- CAGR: {cagr_pct:.2f}%
- Sharpe Ratio: {sharpe_ratio:.3f}
- Sortino Ratio: {sortino_ratio:.3f}
- Max Drawdown: {max_drawdown_pct:.2f}%
- Volatility: {volatility_pct:.2f}%
- Calmar Ratio: {calmar_ratio:.3f}
- Win Rate: {win_rate_pct:.1f}%

**Risk Management Activity**:
- Time in Cash: {time_in_cash_pct:.1f}%
- Market Regime Changes: {regime_change_count}
- Drawdown Protection Triggers: {drawdown_protection_triggers}
- Stop-Loss Activations: {stop_loss_triggers}
- Volatility Breaches: {volatility_breach_triggers}
- Total Risk Events: {total_risk_events}

Create a professional Markdown report following the structure provided in your system prompt.""")
        ])
        
        # Create LCEL chain
        chain = prompt | model | StrOutputParser()
        
        # Prepare input data
        input_data = summary_dict.copy()
        input_data['period_start'] = period_start
        input_data['period_end'] = period_end
        
        # Generate report
        report = chain.invoke(input_data)
        
        # Add header with metadata
        header = f"""---
**Adaptive Portfolio Engine - Comprehensive Analysis Report**  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Period**: {period_start} - {period_end}  
**Market**: NIFTY 50 (Indian Equities)  
**Currency**: INR (₹)

---

"""
        
        return header + report.strip()
    
    except ValueError as e:
        return f"# Report Generation Failed\n\n{str(e)}\n\nPlease configure GEMINI_API_KEY in your .env file."
    except Exception as e:
        return "# Report Generation Failed\n\nAn error occurred while generating the report. Please check your API key configuration."


def generate_stress_test_analysis(
    normal_summary: Dict,
    stressed_summary: Dict,
    stress_type: str
) -> str:
    """
    Generate analysis comparing normal vs stressed backtest results.
    
    Args:
        normal_summary: Summary of normal backtest
        stressed_summary: Summary of stress test backtest
        stress_type: Type of stress test applied
        
    Returns:
        Markdown analysis of stress test results
    """
    try:
        model = get_gemini_model()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial risk analyst evaluating stress test results.

Analyze the impact of stress testing on portfolio performance.

Structure:
# Stress Test Analysis

## Scenario Applied
[Describe the stress scenario]

## Performance Comparison
[Compare normal vs stressed metrics]

## Risk Engine Effectiveness
[Evaluate how well risk controls worked under stress]

## Key Findings
[3-4 bullet points with insights]

## Recommendation
[Brief assessment of resilience]

RULES:
- Be factual and data-driven
- Use markdown formatting
- Compare metrics explicitly
- Focus on risk-adjusted returns"""),
            ("human", """Stress Test Type: {stress_type}

**Normal Conditions**:
- CAGR: {normal_cagr:.2f}%
- Max Drawdown: {normal_dd:.2f}%
- Sharpe: {normal_sharpe:.3f}
- Final Value: ₹{normal_value:,.0f}

**Stressed Conditions** ({stress_type}):
- CAGR: {stressed_cagr:.2f}%
- Max Drawdown: {stressed_dd:.2f}%
- Sharpe: {stressed_sharpe:.3f}
- Final Value: ₹{stressed_value:,.0f}

Analyze the stress test results.""")
        ])
        
        chain = prompt | model | StrOutputParser()
        
        input_data = {
            'stress_type': stress_type,
            'normal_cagr': normal_summary['cagr_pct'],
            'normal_dd': normal_summary['max_drawdown_pct'],
            'normal_sharpe': normal_summary['sharpe_ratio'],
            'normal_value': normal_summary['final_portfolio_value'],
            'stressed_cagr': stressed_summary['cagr_pct'],
            'stressed_dd': stressed_summary['max_drawdown_pct'],
            'stressed_sharpe': stressed_summary['sharpe_ratio'],
            'stressed_value': stressed_summary['final_portfolio_value']
        }
        
        analysis = chain.invoke(input_data)
        return analysis.strip()
    
    except Exception as e:
        return "# Stress Test Analysis Failed\n\nUnable to generate analysis. Please check your API configuration."
