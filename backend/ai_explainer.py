"""
AI Explainer Module using Gemini 2.5 Flash and LangChain LCEL.
Generates short, investor-friendly explanations of backtest results.
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize Gemini model
def get_gemini_model():
    """Initialize Gemini 2.5 Flash model."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY not configured in .env file")
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3  # Low temperature for deterministic, factual responses
    )


def create_summary_dict(backtest_results: Dict, metrics: Dict, risk_logs: list) -> Dict:
    """
    Create a concise summary dictionary from backtest results.
    This prevents sending large arrays to the AI.
    """
    # Count regime changes
    regime_changes = sum(1 for log in risk_logs if 'REGIME_CHANGE' in log.get('event_type', ''))
    
    # Count risk events by type
    drawdown_events = sum(1 for log in risk_logs if 'DRAWDOWN' in log.get('event_type', ''))
    stop_loss_events = sum(1 for log in risk_logs if 'STOP_LOSS' in log.get('event_type', ''))
    vol_breach_events = sum(1 for log in risk_logs if 'VOL_BREACH' in log.get('event_type', ''))
    
    # Get final values
    final_value = backtest_results.get('final_portfolio_value', 100000)
    initial_capital = 100000
    total_return = (final_value - initial_capital) / initial_capital
    
    summary = {
        'initial_capital': initial_capital,
        'final_portfolio_value': final_value,
        'total_return_pct': total_return * 100,
        'cagr_pct': metrics.get('CAGR', 0) * 100,
        'sharpe_ratio': metrics.get('Sharpe Ratio', 0),
        'sortino_ratio': metrics.get('Sortino Ratio', 0),
        'max_drawdown_pct': abs(metrics.get('Max Drawdown', 0) * 100),
        'volatility_pct': metrics.get('Volatility', 0) * 100,
        'calmar_ratio': metrics.get('Calmar Ratio', 0),
        'time_in_cash_pct': metrics.get('Time in Cash (%)', 0),
        'win_rate_pct': metrics.get('Win Rate (%)', 0),
        'regime_change_count': regime_changes,
        'drawdown_protection_triggers': drawdown_events,
        'stop_loss_triggers': stop_loss_events,
        'volatility_breach_triggers': vol_breach_events,
        'total_risk_events': len(risk_logs)
    }
    
    return summary


def generate_explanation(summary_dict: Dict) -> str:
    """
    Generate a short, professional explanation of backtest results using Gemini.
    
    Args:
        summary_dict: Summarized backtest metrics and events
        
    Returns:
        Natural language explanation (2-4 paragraphs)
    """
    try:
        # Initialize model
        model = get_gemini_model()
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional financial risk analyst explaining portfolio backtest results to investors.

Your task is to provide a clear, concise explanation (2-4 paragraphs) covering:
1. Overall performance summary
2. Risk management behavior
3. Key strengths and weaknesses

CRITICAL RULES:
- Use ONLY the provided data - do not invent numbers or statistics
- Be professional but use simple, investor-friendly language
- Avoid technical jargon where possible
- Be honest about both strengths and limitations
- Do not hallucinate or speculate
- Focus on the risk-return tradeoff

Format: Plain text paragraphs, no bullet points."""),
            ("human", """Analyze this portfolio backtest:

Initial Capital: ₹{initial_capital:,.0f}
Final Value: ₹{final_portfolio_value:,.0f}
Total Return: {total_return_pct:.2f}%
CAGR: {cagr_pct:.2f}%
Sharpe Ratio: {sharpe_ratio:.3f}
Sortino Ratio: {sortino_ratio:.3f}
Max Drawdown: {max_drawdown_pct:.2f}%
Volatility: {volatility_pct:.2f}%
Calmar Ratio: {calmar_ratio:.3f}

Risk Management:
- Time in Cash: {time_in_cash_pct:.1f}%
- Regime Changes: {regime_change_count}
- Drawdown Protections Triggered: {drawdown_protection_triggers}
- Stop-Loss Events: {stop_loss_triggers}
- Volatility Breaches: {volatility_breach_triggers}
- Total Risk Events: {total_risk_events}

Provide a professional explanation.""")
        ])
        
        # Create LCEL chain
        chain = prompt | model | StrOutputParser()
        
        # Generate explanation
        explanation = chain.invoke(summary_dict)
        
        return explanation.strip()
    
    except ValueError as e:
        print(f"ValueError in generate_explanation: {str(e)}")
        return f"AI Explanation unavailable: {str(e)}. Please configure GEMINI_API_KEY in your .env file."
    except Exception as e:
        print(f"Exception in generate_explanation: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"AI Explanation unavailable: {type(e).__name__}: {str(e)}"


def generate_quick_insight(summary_dict: Dict, question: str) -> str:
    """
    Generate a contextual answer to a specific question about the backtest.
    
    Args:
        summary_dict: Summarized backtest metrics
        question: User's question
        
    Returns:
        Contextual answer based on provided data
    """
    try:
        model = get_gemini_model()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional financial analyst answering specific questions about portfolio backtest results.

CRITICAL RULES:
- Answer ONLY based on the provided data
- If the data doesn't contain information to answer the question, say so
- Do not invent statistics or speculate
- Be concise (2-3 sentences maximum)
- Be factual and professional"""),
            ("human", """Portfolio Data:
CAGR: {cagr_pct:.2f}%
Max Drawdown: {max_drawdown_pct:.2f}%
Sharpe Ratio: {sharpe_ratio:.3f}
Time in Cash: {time_in_cash_pct:.1f}%
Regime Changes: {regime_change_count}
Risk Events: {total_risk_events}

Question: {question}

Answer based only on the data provided:""")
        ])
        
        # Create LCEL chain
        chain = prompt | model | StrOutputParser()
        
        # Generate answer
        input_data = summary_dict.copy()
        input_data['question'] = question
        answer = chain.invoke(input_data)
        
        return answer.strip()
    
    except Exception as e:
        return "Unable to generate answer. Please check your API configuration."
