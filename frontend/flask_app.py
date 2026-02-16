"""
Flask Frontend for Adaptive Portfolio Engine
Beautiful, professional web interface using Jinja templates and Bootstrap 5.
"""

import os
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
from flask_session import Session
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import markdown
import io

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure server-side session to handle large data
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=5)

# Initialize the session
Session(app)

# Backend API URL
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')


# ============= Helper Functions =============

def call_backend_api(endpoint, method='GET', data=None, params=None):
    """Make API call to FastAPI backend."""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        if method == 'GET':
            response = requests.get(url, params=params, timeout=600)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=600)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


def format_currency(value):
    """Format value as Indian Rupees."""
    return f"₹{value:,.0f}"


def format_percentage(value):
    """Format value as percentage."""
    return f"{value:.2f}%"


# ============= Routes =============

@app.route('/')
def index():
    """Home page with backtest controls."""
    return render_template('index.html')


@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """Run backtest and display results."""
    start_year = int(request.form.get('start_year', 2015))
    end_year = int(request.form.get('end_year', 2024))
    with_risk_engine = request.form.get('with_risk_engine') == 'on'
    
    # Call backend
    result = call_backend_api(
        '/run_backtest',
        method='POST',
        data={
            'start_year': start_year,
            'end_year': end_year,
            'with_risk_engine': with_risk_engine
        }
    )
    
    if 'error' in result:
        return render_template('error.html', error=result['error'])
    
    # Store in session for AI generation (make session permanent)
    session.permanent = True
    session['last_backtest'] = result
    session['start_year'] = start_year
    session['end_year'] = end_year
    
    return render_template('results.html', 
                         results=result, 
                         start_year=start_year,
                         end_year=end_year,
                         with_risk_engine=with_risk_engine)


@app.route('/run_stress_test', methods=['POST'])
def run_stress_test():
    """Run stress test and display results."""
    start_year = int(request.form.get('start_year', 2015))
    end_year = int(request.form.get('end_year', 2024))
    with_risk_engine = request.form.get('with_risk_engine') == 'on'
    stress_type = request.form.get('stress_type', 'market_shock')
    
    # If stress_type is empty string, default to market_shock
    if not stress_type or stress_type == '':
        stress_type = 'market_shock'
    
    # Call backend
    result = call_backend_api(
        '/run_stress_test',
        method='POST',
        data={
            'start_year': start_year,
            'end_year': end_year,
            'with_risk_engine': with_risk_engine,
            'stress_type': stress_type
        }
    )
    
    if 'error' in result:
        return render_template('error.html', error=result['error'])
    
    # Store in session (make session permanent)
    session.permanent = True
    session['last_backtest'] = result
    session['start_year'] = start_year
    session['end_year'] = end_year
    
    return render_template('results.html', 
                         results=result, 
                         start_year=start_year,
                         end_year=end_year,
                         with_risk_engine=with_risk_engine,
                         is_stress=True,
                         stress_type=stress_type)


@app.route('/generate_ai_explanation', methods=['POST'])
def generate_ai_explanation():
    """Generate AI explanation for last backtest."""
    if 'last_backtest' not in session:
        return jsonify({'error': 'No backtest results found. Please run a backtest first.'})
    
    results = session['last_backtest']
    metrics = results['metrics']
    risk_logs = results['risk_logs']
    
    # Create summary dict
    summary_dict = {
        'initial_capital': 100000,
        'final_portfolio_value': results.get('total_portfolio_value', 100000),
        'total_return_pct': (results.get('total_portfolio_value', 100000) - 100000) / 1000,
        'cagr_pct': metrics.get('CAGR', 0) * 100,
        'sharpe_ratio': metrics.get('Sharpe Ratio', 0),
        'sortino_ratio': metrics.get('Sortino Ratio', 0),
        'max_drawdown_pct': abs(metrics.get('Max Drawdown', 0) * 100),
        'volatility_pct': metrics.get('Volatility', 0) * 100,
        'calmar_ratio': metrics.get('Calmar Ratio', 0),
        'time_in_cash_pct': metrics.get('Time in Cash (%)', 0),
        'win_rate_pct': metrics.get('Win Rate (%)', 0),
        'regime_change_count': sum(1 for log in risk_logs if 'REGIME_CHANGE' in log.get('event_type', '')),
        'drawdown_protection_triggers': sum(1 for log in risk_logs if 'DRAWDOWN' in log.get('event_type', '')),
        'stop_loss_triggers': sum(1 for log in risk_logs if 'STOP_LOSS' in log.get('event_type', '')),
        'volatility_breach_triggers': sum(1 for log in risk_logs if 'VOL_BREACH' in log.get('event_type', '')),
        'total_risk_events': len(risk_logs)
    }
    
    # Call AI API
    ai_result = call_backend_api(
        '/generate_explanation',
        method='POST',
        data={'backtest_summary': summary_dict}
    )
    
    return jsonify(ai_result)


@app.route('/generate_ai_report', methods=['POST'])
def generate_ai_report():
    """Generate full AI report for last backtest."""
    if 'last_backtest' not in session:
        return jsonify({'error': 'No backtest results found. Please run a backtest first.'})
    
    results = session['last_backtest']
    metrics = results['metrics']
    risk_logs = results['risk_logs']
    
    # Create summary dict
    summary_dict = {
        'initial_capital': 100000,
        'final_portfolio_value': results.get('total_portfolio_value', 100000),
        'total_return_pct': (results.get('total_portfolio_value', 100000) - 100000) / 1000,
        'cagr_pct': metrics.get('CAGR', 0) * 100,
        'sharpe_ratio': metrics.get('Sharpe Ratio', 0),
        'sortino_ratio': metrics.get('Sortino Ratio', 0),
        'max_drawdown_pct': abs(metrics.get('Max Drawdown', 0) * 100),
        'volatility_pct': metrics.get('Volatility', 0) * 100,
        'calmar_ratio': metrics.get('Calmar Ratio', 0),
        'time_in_cash_pct': metrics.get('Time in Cash (%)', 0),
        'win_rate_pct': metrics.get('Win Rate (%)', 0),
        'regime_change_count': sum(1 for log in risk_logs if 'REGIME_CHANGE' in log.get('event_type', '')),
        'drawdown_protection_triggers': sum(1 for log in risk_logs if 'DRAWDOWN' in log.get('event_type', '')),
        'stop_loss_triggers': sum(1 for log in risk_logs if 'STOP_LOSS' in log.get('event_type', '')),
        'volatility_breach_triggers': sum(1 for log in risk_logs if 'VOL_BREACH' in log.get('event_type', '')),
        'total_risk_events': len(risk_logs)
    }
    
    # Call AI API
    ai_result = call_backend_api(
        '/generate_full_report',
        method='POST',
        data={
            'backtest_summary': summary_dict,
            'period_start': str(session.get('start_year', 2015)),
            'period_end': str(session.get('end_year', 2024))
        }
    )
    
    # Convert markdown to HTML if report exists
    if 'report' in ai_result:
        ai_result['report_html'] = markdown.markdown(
            ai_result['report'],
            extensions=['tables', 'fenced_code', 'nl2br']
        )
    
    return jsonify(ai_result)


@app.route('/download_report/<format_type>')
def download_report(format_type):
    """Download AI report as file."""
    if 'ai_report' not in session:
        return "No report available", 404
    
    report_text = session['ai_report']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format_type == 'md':
        filename = f'portfolio_report_{timestamp}.md'
        mimetype = 'text/markdown'
    else:
        filename = f'portfolio_report_{timestamp}.txt'
        mimetype = 'text/plain'
    
    # Create in-memory file
    file_obj = io.BytesIO(report_text.encode('utf-8'))
    file_obj.seek(0)
    
    return send_file(
        file_obj,
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename
    )


@app.route('/compare_scenarios', methods=['POST'])
def compare_scenarios():
    """Compare all 4 scenarios (normal/stressed × with/without risk)."""
    try:
        # Get parameters from form or session
        start_year = int(request.form.get('start_year', session.get('start_year', 2015)))
        end_year = int(request.form.get('end_year', session.get('end_year', 2024)))
        
        # Call backend API
        response = call_backend_api(
            f'/compare_risk_engine?start_year={start_year}&end_year={end_year}',
            method='GET'
        )
        
        if 'error' in response:
            return render_template('error.html', error=response['error'])
        
        if response and 'comparison' in response:
            # Store in session
            session.permanent = True
            session['comparison_results'] = response
            session['start_year'] = start_year
            session['end_year'] = end_year
            return render_template('comparison.html', data=response, start_year=start_year, end_year=end_year)
        else:
            return render_template('error.html', error="Failed to retrieve comparison results")
    
    except Exception as e:
        return render_template('error.html', error=str(e))


# ============= Template Filters =============

@app.template_filter('currency')
def currency_filter(value):
    """Jinja filter for currency formatting."""
    return format_currency(value)


@app.template_filter('percentage')
def percentage_filter(value):
    """Jinja filter for percentage formatting."""
    return format_percentage(value * 100)


@app.template_filter('markdown_to_html')
def markdown_filter(text):
    """Convert markdown to HTML."""
    return markdown.markdown(text)


# ============= Run App =============

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Flask Portfolio Dashboard")
    print("="*60)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Dashboard URL: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, threaded=True)
