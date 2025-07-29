import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sqlite3
import warnings
import ssl
import urllib3

# Disable SSL warnings and verification issues
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create unverified SSL context to handle certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# Page configuration
st.set_page_config(
    page_title="Trading Performance Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database functions
@st.cache_resource
def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect('trading_performance.db', check_same_thread=False)
    
    # Create trades table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS daily_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE UNIQUE NOT NULL,
            opening_balance REAL NOT NULL,
            closing_balance REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create benchmarks table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            symbol TEXT NOT NULL,
            close_price REAL NOT NULL,
            daily_return REAL,
            UNIQUE(date, symbol)
        )
    ''')
    
    conn.commit()
    return conn

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_benchmark_data(symbols, start_date, end_date):
    """Fetch benchmark data from yfinance with robust error handling"""
    benchmark_data = {}
    
    for symbol in symbols:
        try:
            # Method 1: Try bulk download first (same as your working dashboard)
            try:
                data = yf.download(symbol, start=start_date, end=end_date, 
                                 progress=False, show_errors=False, threads=True)
                
                if not data.empty and 'Adj Close' in data.columns:
                    data['daily_return'] = data['Adj Close'].pct_change()
                    result_data = data[['Adj Close', 'daily_return']].reset_index()
                    result_data = result_data.rename(columns={'Adj Close': 'Close'})
                    result_data['Date'] = result_data['Date'].dt.date
                    benchmark_data[symbol] = result_data
                    st.success(f"✅ Successfully fetched data for {symbol}")
                    continue
            except Exception as e1:
                st.warning(f"Bulk download failed for {symbol}, trying individual method...")
            
            # Method 2: Individual ticker method (fallback)
            ticker_obj = yf.Ticker(symbol)
            ticker_data = ticker_obj.history(start=start_date, end=end_date, 
                                           auto_adjust=True, progress=False)
            
            if not ticker_data.empty and 'Close' in ticker_data.columns:
                ticker_data['daily_return'] = ticker_data['Close'].pct_change()
                result_data = ticker_data[['Close', 'daily_return']].reset_index()
                result_data['Date'] = result_data['Date'].dt.date
                benchmark_data[symbol] = result_data
                st.success(f"✅ Successfully fetched data for {symbol} (individual method)")
            else:
                st.warning(f"No data returned for {symbol}")
                
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            continue
    
    return benchmark_data

def save_benchmark_data(conn, benchmark_data):
    """Save benchmark data to database"""
    for symbol, data in benchmark_data.items():
        for _, row in data.iterrows():
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO benchmarks (date, symbol, close_price, daily_return)
                    VALUES (?, ?, ?, ?)
                ''', (row['Date'], symbol, row['Close'], row['daily_return']))
            except Exception as e:
                st.error(f"Error saving benchmark data: {e}")
    conn.commit()

def load_trading_data(conn):
    """Load trading data from database"""
    query = "SELECT * FROM daily_trades ORDER BY date"
    return pd.read_sql_query(query, conn, parse_dates=['date'])

def load_benchmark_data(conn, symbols):
    """Load benchmark data from database"""
    if not symbols:
        return pd.DataFrame()
    
    placeholders = ','.join('?' * len(symbols))
    query = f"SELECT * FROM benchmarks WHERE symbol IN ({placeholders}) ORDER BY date"
    return pd.read_sql_query(query, conn, params=symbols, parse_dates=['date'])

# Performance calculation functions
def calculate_performance_metrics(df):
    """Calculate comprehensive performance metrics"""
    if df.empty or len(df) < 2:
        return {}
    
    # Basic calculations
    df = df.sort_values('date').reset_index(drop=True)
    df['daily_return'] = df['realized_pnl'] / df['opening_balance']
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
    df['equity_curve'] = df['opening_balance'].iloc[0] * (1 + df['cumulative_return'])
    
    # Calculate drawdown
    df['peak'] = df['equity_curve'].expanding().max()
    df['drawdown'] = (df['equity_curve'] - df['peak']) / df['peak']
    
    # Performance metrics
    total_days = len(df)
    trading_days = len(df[df['realized_pnl'] != 0])
    winning_days = len(df[df['realized_pnl'] > 0])
    losing_days = len(df[df['realized_pnl'] < 0])
    
    total_return = df['cumulative_return'].iloc[-1]
    daily_returns = df['daily_return'].dropna()
    
    # Risk metrics
    volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
    sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0
    
    # Downside deviation for Sortino ratio
    downside_returns = daily_returns[daily_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
    sortino_ratio = (daily_returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0
    
    max_drawdown = df['drawdown'].min()
    calmar_ratio = (daily_returns.mean() * 252) / abs(max_drawdown) if max_drawdown < 0 else 0
    
    # Win/Loss streaks
    df['win_loss'] = np.where(df['realized_pnl'] > 0, 1, np.where(df['realized_pnl'] < 0, -1, 0))
    streaks = calculate_streaks(df['win_loss'].values)
    
    return {
        'total_return': total_return,
        'total_pnl': df['realized_pnl'].sum(),
        'total_days': total_days,
        'trading_days': trading_days,
        'winning_days': winning_days,
        'losing_days': losing_days,
        'win_rate': winning_days / trading_days if trading_days > 0 else 0,
        'avg_win': df[df['realized_pnl'] > 0]['realized_pnl'].mean() if winning_days > 0 else 0,
        'avg_loss': df[df['realized_pnl'] < 0]['realized_pnl'].mean() if losing_days > 0 else 0,
        'profit_factor': abs(df[df['realized_pnl'] > 0]['realized_pnl'].sum() / 
                            df[df['realized_pnl'] < 0]['realized_pnl'].sum()) if losing_days > 0 else float('inf'),
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'max_winning_streak': streaks['max_winning_streak'],
        'max_losing_streak': streaks['max_losing_streak'],
        'current_streak': streaks['current_streak']
    }

def calculate_period_performance(df):
    """Calculate weekly, monthly, and quarterly performance"""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    df = df.sort_values('date').copy()
    df['daily_return'] = df['realized_pnl'] / df['opening_balance']
    
    # Weekly performance
    df['week'] = df['date'].dt.to_period('W')
    weekly_perf = df.groupby('week').agg({
        'realized_pnl': 'sum',
        'daily_return': lambda x: (1 + x).prod() - 1,
        'opening_balance': 'first'
    }).reset_index()
    weekly_perf['week'] = weekly_perf['week'].astype(str)
    weekly_perf['return_pct'] = weekly_perf['daily_return']
    weekly_perf = weekly_perf.rename(columns={'week': 'period'})
    
    # Monthly performance
    df['month'] = df['date'].dt.to_period('M')
    monthly_perf = df.groupby('month').agg({
        'realized_pnl': 'sum',
        'daily_return': lambda x: (1 + x).prod() - 1,
        'opening_balance': 'first'
    }).reset_index()
    monthly_perf['month'] = monthly_perf['month'].astype(str)
    monthly_perf['return_pct'] = monthly_perf['daily_return']
    monthly_perf = monthly_perf.rename(columns={'month': 'period'})
    
    # Quarterly performance
    df['quarter'] = df['date'].dt.to_period('Q')
    quarterly_perf = df.groupby('quarter').agg({
        'realized_pnl': 'sum',
        'daily_return': lambda x: (1 + x).prod() - 1,
        'opening_balance': 'first'
    }).reset_index()
    quarterly_perf['quarter'] = quarterly_perf['quarter'].astype(str)
    quarterly_perf['return_pct'] = quarterly_perf['daily_return']
    quarterly_perf = quarterly_perf.rename(columns={'quarter': 'period'})
    
    return weekly_perf, monthly_perf, quarterly_perf

def calculate_rolling_metrics(df, windows=[30, 60, 90]):
    """Calculate rolling performance metrics"""
    if df.empty or len(df) < max(windows):
        return pd.DataFrame()
    
    df = df.sort_values('date').copy()
    df['daily_return'] = df['realized_pnl'] / df['opening_balance']
    
    rolling_metrics = pd.DataFrame()
    rolling_metrics['date'] = df['date']
    
    for window in windows:
        if len(df) >= window:
            # Rolling returns
            rolling_metrics[f'return_{window}d'] = df['daily_return'].rolling(window=window).apply(
                lambda x: (1 + x).prod() - 1
            )
            
            # Rolling Sharpe ratio
            rolling_metrics[f'sharpe_{window}d'] = df['daily_return'].rolling(window=window).apply(
                lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
            )
            
            # Rolling volatility
            rolling_metrics[f'volatility_{window}d'] = df['daily_return'].rolling(window=window).std() * np.sqrt(252)
            
            # Rolling max drawdown
            equity_rolling = df['daily_return'].rolling(window=window).apply(
                lambda x: (1 + x).cumprod()
            )
            rolling_metrics[f'max_dd_{window}d'] = equity_rolling.rolling(window=window).apply(
                lambda x: ((x - x.expanding().max()) / x.expanding().max()).min()
            )
            
            # Rolling win rate
            rolling_metrics[f'win_rate_{window}d'] = (df['realized_pnl'] > 0).rolling(window=window).mean()
    
    return rolling_metrics.dropna()

def calculate_streaks(win_loss_array):
    """Calculate winning and losing streaks"""
    if len(win_loss_array) == 0:
        return {'max_winning_streak': 0, 'max_losing_streak': 0, 'current_streak': 0}
    
    max_winning_streak = 0
    max_losing_streak = 0
    current_streak = 0
    current_streak_type = 0
    
    for val in win_loss_array:
        if val == 1:  # Win
            if current_streak_type == 1:
                current_streak += 1
            else:
                current_streak = 1
                current_streak_type = 1
            max_winning_streak = max(max_winning_streak, current_streak)
        elif val == -1:  # Loss
            if current_streak_type == -1:
                current_streak += 1
            else:
                current_streak = 1
                current_streak_type = -1
            max_losing_streak = max(max_losing_streak, current_streak)
        else:  # No trade
            current_streak = 0
            current_streak_type = 0
    
    return {
        'max_winning_streak': max_winning_streak,
        'max_losing_streak': max_losing_streak,
        'current_streak': current_streak if current_streak_type != 0 else 0
    }

def calculate_benchmark_comparison(trading_df, benchmark_df, benchmark_symbols):
    """Calculate performance vs benchmarks"""
    if trading_df.empty or benchmark_df.empty:
        return {}
    
    # Align dates
    trading_df = trading_df.set_index('date')['daily_return']
    
    comparisons = {}
    for symbol in benchmark_symbols:
        bench_data = benchmark_df[benchmark_df['symbol'] == symbol].set_index('date')['daily_return']
        
        # Align dates
        aligned_data = pd.concat([trading_df, bench_data], axis=1, join='inner')
        aligned_data.columns = ['trading', 'benchmark']
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) > 1:
            # Calculate correlation and beta
            correlation = aligned_data['trading'].corr(aligned_data['benchmark'])
            covariance = aligned_data['trading'].cov(aligned_data['benchmark'])
            benchmark_var = aligned_data['benchmark'].var()
            beta = covariance / benchmark_var if benchmark_var > 0 else 0
            
            # Calculate alpha (using simple CAPM)
            risk_free_rate = 0.02 / 252  # Assume 2% annual risk-free rate
            trading_excess = aligned_data['trading'].mean() - risk_free_rate
            benchmark_excess = aligned_data['benchmark'].mean() - risk_free_rate
            alpha = trading_excess - beta * benchmark_excess
            
            comparisons[symbol] = {
                'correlation': correlation,
                'beta': beta,
                'alpha': alpha * 252,  # Annualized
                'tracking_error': (aligned_data['trading'] - aligned_data['benchmark']).std() * np.sqrt(252)
            }
    
    return comparisons

# Streamlit UI
def main():
    st.title("📈 Trading Performance Dashboard")
    
    # Initialize database
    conn = init_database()
    
    # Sidebar for data entry and settings
    with st.sidebar:
        st.header("Settings & Data Entry")
        
        # Benchmark symbols
        st.subheader("Settings")
        
        # Add any other settings here if needed
        st.info("Dashboard settings and controls")
        
        # Data entry form
        st.subheader("Add Trading Day")
        with st.form("trade_entry"):
            trade_date = st.date_input("Date", value=datetime.now().date())
            opening_balance = st.number_input("Opening Balance ($)", min_value=0.0, step=100.0)
            closing_balance = st.number_input("Closing Balance ($)", min_value=0.0, step=100.0)
            realized_pnl = st.number_input("Realized P&L ($)", step=10.0)
            notes = st.text_area("Notes (optional)")
            
            if st.form_submit_button("Add Entry"):
                try:
                    conn.execute('''
                        INSERT OR REPLACE INTO daily_trades (date, opening_balance, closing_balance, realized_pnl, notes)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (trade_date, opening_balance, closing_balance, realized_pnl, notes))
                    conn.commit()
                    st.success("Trade entry added successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding entry: {e}")
        
        # Delete entries section
        st.subheader("Manage Entries")
        
        # Load current data for deletion
        current_data = load_trading_data(conn)
        if not current_data.empty:
            st.write("**Delete Entries:**")
            dates_to_delete = st.multiselect(
                "Select dates to delete:",
                options=current_data['date'].dt.strftime('%Y-%m-%d').tolist(),
                help="Select one or more dates to delete"
            )
            
            if dates_to_delete:
                if st.button("🗑️ Delete Selected Entries", type="secondary"):
                    try:
                        for date_str in dates_to_delete:
                            conn.execute('DELETE FROM daily_trades WHERE date = ?', (date_str,))
                        conn.commit()
                        st.success(f"Successfully deleted {len(dates_to_delete)} entries!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting entries: {e}")
            
            # Clear all data option
            if st.button("🗑️ Clear All Data", type="secondary"):
                if st.checkbox("I confirm I want to delete ALL trading data"):
                    try:
                        conn.execute('DELETE FROM daily_trades')
                        conn.commit()
                        st.success("All trading data cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing data: {e}")
        else:
            st.info("No entries to delete")
    
    # Load data
    trading_data = load_trading_data(conn)
    
    if trading_data.empty:
        st.warning("No trading data found. Please add some entries using the sidebar.")
        return
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(trading_data)
    
    # Calculate period performance and rolling metrics
    weekly_perf, monthly_perf, quarterly_perf = calculate_period_performance(trading_data)
    rolling_metrics = calculate_rolling_metrics(trading_data)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Detailed Analytics", "📅 Period Performance", "📋 Trade History"])
    
    with tab1:
        # Overview metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
            st.metric("Total P&L", f"${metrics.get('total_pnl', 0):,.2f}")
        
        with col2:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
            st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
        
        with col4:
            current_streak = metrics.get('current_streak', 0)
            streak_label = f"Current Streak: {abs(current_streak)} {'Wins' if current_streak > 0 else 'Losses' if current_streak < 0 else 'None'}"
            st.metric("Volatility", f"{metrics.get('volatility', 0):.1%}")
            st.write(streak_label)
        
        # Equity curve chart
        st.subheader("Cumulative Return")
        
        if len(trading_data) > 1:
            fig = go.Figure()
            
            # Add trading cumulative return curve
            trading_data_sorted = trading_data.sort_values('date')
            trading_data_sorted['daily_return'] = trading_data_sorted['realized_pnl'] / trading_data_sorted['opening_balance']
            trading_data_sorted['cumulative_return'] = (1 + trading_data_sorted['daily_return']).cumprod() - 1
            
            fig.add_trace(go.Scatter(
                x=trading_data_sorted['date'],
                y=trading_data_sorted['cumulative_return'] * 100,  # Convert to percentage
                name='Trading Account',
                line=dict(color='#1f77b4', width=3)
            ))
            
            fig.update_layout(
                title="Cumulative Return Over Time",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                hovermode='x unified',
                height=500
            )
            
            # Format y-axis as percentage
            fig.update_yaxes(tickformat='.1f', ticksuffix='%')
            
            st.plotly_chart(fig, use_container_width=True)
        

    
    with tab2:
        st.header("Detailed Analytics")
        
        # Risk-adjusted metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk-Adjusted Performance")
            risk_metrics = pd.DataFrame({
                'Metric': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'],
                'Value': [
                    metrics.get('sharpe_ratio', 0),
                    metrics.get('sortino_ratio', 0),
                    metrics.get('calmar_ratio', 0)
                ]
            })
            st.dataframe(risk_metrics, hide_index=True)
            
            st.subheader("Streak Analysis")
            streak_data = pd.DataFrame({
                'Streak Type': ['Max Winning', 'Max Losing', 'Current'],
                'Days': [
                    metrics.get('max_winning_streak', 0),
                    metrics.get('max_losing_streak', 0),
                    abs(metrics.get('current_streak', 0))
                ]
            })
            st.dataframe(streak_data, hide_index=True)
        
        with col2:
            st.subheader("Benchmark Comparison")
            st.info("Benchmark comparison has been removed for simplicity")
        
        # Drawdown analysis
        if len(trading_data) > 1:
            st.subheader("Drawdown Analysis")
            
            trading_data_sorted = trading_data.sort_values('date')
            trading_data_sorted['daily_return'] = trading_data_sorted['realized_pnl'] / trading_data_sorted['opening_balance']
            trading_data_sorted['cumulative_return'] = (1 + trading_data_sorted['daily_return']).cumprod() - 1
            trading_data_sorted['equity_curve'] = trading_data_sorted['opening_balance'].iloc[0] * (1 + trading_data_sorted['cumulative_return'])
            trading_data_sorted['peak'] = trading_data_sorted['equity_curve'].expanding().max()
            trading_data_sorted['drawdown'] = (trading_data_sorted['equity_curve'] - trading_data_sorted['peak']) / trading_data_sorted['peak']
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Equity Curve', 'Drawdown'),
                vertical_spacing=0.1
            )
            
            # Equity curve
            fig.add_trace(
                go.Scatter(
                    x=trading_data_sorted['date'],
                    y=trading_data_sorted['equity_curve'],
                    name='Equity',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=trading_data_sorted['date'],
                    y=trading_data_sorted['peak'],
                    name='Peak',
                    line=dict(color='green', dash='dash')
                ),
                row=1, col=1
            )
            
            # Drawdown
            fig.add_trace(
                go.Scatter(
                    x=trading_data_sorted['date'],
                    y=trading_data_sorted['drawdown'],
                    name='Drawdown',
                    fill='tonexty',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=True)
            fig.update_yaxes(title_text="Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", tickformat='.1%', row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Return distribution
        if len(trading_data) > 1:
            st.subheader("Return Distribution")
            
            returns = trading_data['realized_pnl'] / trading_data['opening_balance']
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=20,
                name='Daily Returns',
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Distribution of Daily Returns",
                xaxis_title="Daily Return",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Period Performance & Rolling Metrics")
        
        # Period Performance Section
        st.subheader("Performance by Time Period")
        
        # Create tabs for different periods
        period_tab1, period_tab2, period_tab3, period_tab4 = st.tabs(["📅 Daily", "📅 Weekly", "📅 Monthly", "📅 Quarterly"])
        
        with period_tab1:
            if not trading_data.empty:
                st.write("**Daily Performance**")
                
                # Calculate daily performance data
                daily_data = trading_data.copy().sort_values('date')
                daily_data['daily_return_pct'] = (daily_data['realized_pnl'] / daily_data['opening_balance'])  # Keep as decimal
                daily_data['cumulative_pnl'] = daily_data['realized_pnl'].cumsum()
                
                # Display daily performance table
                daily_display = daily_data.copy()
                daily_display['date'] = daily_display['date'].dt.strftime('%Y-%m-%d')
                daily_display['opening_balance'] = daily_display['opening_balance'].apply(lambda x: f"${x:,.2f}")
                daily_display['closing_balance'] = daily_display['closing_balance'].apply(lambda x: f"${x:,.2f}")
                daily_display['realized_pnl'] = daily_display['realized_pnl'].apply(lambda x: f"${x:,.2f}")
                daily_display['daily_return_pct'] = daily_display['daily_return_pct'].apply(lambda x: f"{x:.2%}")  # Format as percentage here
                daily_display['cumulative_pnl'] = daily_display['cumulative_pnl'].apply(lambda x: f"${x:,.2f}")
                
                daily_display = daily_display[['date', 'opening_balance', 'closing_balance', 'realized_pnl', 'daily_return_pct', 'cumulative_pnl']].rename(columns={
                    'date': 'Date',
                    'opening_balance': 'Opening Balance',
                    'closing_balance': 'Closing Balance', 
                    'realized_pnl': 'Realized P&L',
                    'daily_return_pct': 'Daily Return %',
                    'cumulative_pnl': 'Cumulative P&L'
                })
                
                # Reverse to show most recent first
                daily_display = daily_display.iloc[::-1]
                
                st.dataframe(daily_display, hide_index=True, use_container_width=True)
                
                # Daily returns chart
                fig = go.Figure()
                
                # Add daily returns as bars (convert to percentage for display)
                daily_returns = (trading_data['realized_pnl'] / trading_data['opening_balance']) * 100  # Convert to percentage for chart
                colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in daily_returns]
                
                fig.add_trace(go.Bar(
                    x=trading_data['date'],
                    y=daily_returns,
                    name='Daily Return %',
                    marker_color=colors,
                    text=[f"{x:.1f}%" for x in daily_returns],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Daily Returns (%)",
                    xaxis_title="Date",
                    yaxis_title="Return (%)",
                    height=400
                )
                
                # Add horizontal line at 0%
                fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Daily performance summary stats
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate stats using decimal values, then format as percentages
                daily_returns_decimal = trading_data['realized_pnl'] / trading_data['opening_balance']
                
                with col1:
                    avg_daily_return = daily_returns_decimal.mean()
                    st.metric("Avg Daily Return", f"{avg_daily_return:.2%}")
                
                with col2:
                    positive_days = len(daily_returns_decimal[daily_returns_decimal > 0])
                    total_days = len(daily_returns_decimal[daily_returns_decimal != 0])
                    win_rate = positive_days / total_days if total_days > 0 else 0
                    st.metric("Daily Win Rate", f"{win_rate:.1%}")
                
                with col3:
                    best_day = daily_returns_decimal.max() if len(daily_returns_decimal) > 0 else 0
                    st.metric("Best Day", f"{best_day:.2%}")
                
                with col4:
                    worst_day = daily_returns_decimal.min() if len(daily_returns_decimal) > 0 else 0
                    st.metric("Worst Day", f"{worst_day:.2%}")
            else:
                st.info("No daily data available")
        
        with period_tab4:
            if not weekly_perf.empty:
                st.write("**Weekly Performance**")
                
                # Display weekly performance table
                weekly_display = weekly_perf.copy()
                weekly_display['realized_pnl'] = weekly_display['realized_pnl'].apply(lambda x: f"${x:,.2f}")
                weekly_display['return_pct'] = weekly_display['return_pct'].apply(lambda x: f"{x:.2%}")
                weekly_display = weekly_display[['period', 'realized_pnl', 'return_pct']].rename(columns={
                    'period': 'Week',
                    'realized_pnl': 'P&L',
                    'return_pct': 'Return %'
                })
                
                st.dataframe(weekly_display, hide_index=True, use_container_width=True)
                
                # Weekly performance chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=weekly_perf['period'],
                    y=weekly_perf['return_pct'] * 100,
                    name='Weekly Return %',
                    marker_color=['green' if x > 0 else 'red' for x in weekly_perf['return_pct']]
                ))
                
                fig.update_layout(
                    title="Weekly Returns (%)",
                    xaxis_title="Week",
                    yaxis_title="Return (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No weekly data available")
        
        with period_tab2:
            if not monthly_perf.empty:
                st.write("**Monthly Performance**")
                
                # Display monthly performance table
                monthly_display = monthly_perf.copy()
                monthly_display['realized_pnl'] = monthly_display['realized_pnl'].apply(lambda x: f"${x:,.2f}")
                monthly_display['return_pct'] = monthly_display['return_pct'].apply(lambda x: f"{x:.2%}")
                monthly_display = monthly_display[['period', 'realized_pnl', 'return_pct']].rename(columns={
                    'period': 'Month',
                    'realized_pnl': 'P&L',
                    'return_pct': 'Return %'
                })
                
                st.dataframe(monthly_display, hide_index=True, use_container_width=True)
                
                # Monthly performance chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=monthly_perf['period'],
                    y=monthly_perf['return_pct'] * 100,
                    name='Monthly Return %',
                    marker_color=['green' if x > 0 else 'red' for x in monthly_perf['return_pct']]
                ))
                
                fig.update_layout(
                    title="Monthly Returns (%)",
                    xaxis_title="Month",
                    yaxis_title="Return (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No monthly data available")
        
        with period_tab3:
            if not quarterly_perf.empty:
                st.write("**Quarterly Performance**")
                
                # Display quarterly performance table
                quarterly_display = quarterly_perf.copy()
                quarterly_display['realized_pnl'] = quarterly_display['realized_pnl'].apply(lambda x: f"${x:,.2f}")
                quarterly_display['return_pct'] = quarterly_display['return_pct'].apply(lambda x: f"{x:.2%}")
                quarterly_display = quarterly_display[['period', 'realized_pnl', 'return_pct']].rename(columns={
                    'period': 'Quarter',
                    'realized_pnl': 'P&L',
                    'return_pct': 'Return %'
                })
                
                st.dataframe(quarterly_display, hide_index=True, use_container_width=True)
                
                # Quarterly performance chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=quarterly_perf['period'],
                    y=quarterly_perf['return_pct'] * 100,
                    name='Quarterly Return %',
                    marker_color=['green' if x > 0 else 'red' for x in quarterly_perf['return_pct']]
                ))
                
                fig.update_layout(
                    title="Quarterly Returns (%)",
                    xaxis_title="Quarter",
                    yaxis_title="Return (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No quarterly data available")
        
        # Rolling Metrics Section
        st.subheader("Rolling Performance Metrics")
        
        if not rolling_metrics.empty:
            # Rolling metrics selection
            col1, col2 = st.columns(2)
            
            with col1:
                metric_options = ['return', 'sharpe', 'volatility', 'max_dd', 'win_rate']
                selected_metric = st.selectbox(
                    "Select Metric to Display:",
                    options=metric_options,
                    format_func=lambda x: {
                        'return': 'Rolling Returns',
                        'sharpe': 'Rolling Sharpe Ratio',
                        'volatility': 'Rolling Volatility',
                        'max_dd': 'Rolling Max Drawdown',
                        'win_rate': 'Rolling Win Rate'
                    }[x]
                )
            
            with col2:
                window_options = [30, 60, 90]
                selected_windows = st.multiselect(
                    "Select Time Windows (days):",
                    options=window_options,
                    default=window_options
                )
            
            if selected_windows:
                # Create rolling metrics chart
                fig = go.Figure()
                
                for window in selected_windows:
                    col_name = f'{selected_metric}_{window}d'
                    if col_name in rolling_metrics.columns:
                        y_values = rolling_metrics[col_name]
                        
                        # Format y-values based on metric type
                        if selected_metric in ['return', 'volatility', 'max_dd', 'win_rate']:
                            y_values = y_values * 100  # Convert to percentage
                        
                        fig.add_trace(go.Scatter(
                            x=rolling_metrics['date'],
                            y=y_values,
                            name=f'{window}-day',
                            mode='lines',
                            line=dict(width=2)
                        ))
                
                # Update layout based on metric type
                title_map = {
                    'return': 'Rolling Returns (%)',
                    'sharpe': 'Rolling Sharpe Ratio',
                    'volatility': 'Rolling Volatility (%)',
                    'max_dd': 'Rolling Max Drawdown (%)',
                    'win_rate': 'Rolling Win Rate (%)'
                }
                
                y_axis_map = {
                    'return': 'Return (%)',
                    'sharpe': 'Sharpe Ratio',
                    'volatility': 'Volatility (%)',
                    'max_dd': 'Max Drawdown (%)',
                    'win_rate': 'Win Rate (%)'
                }
                
                fig.update_layout(
                    title=title_map[selected_metric],
                    xaxis_title="Date",
                    yaxis_title=y_axis_map[selected_metric],
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Rolling metrics summary table
                st.write("**Rolling Metrics Summary (Latest Values)**")
                
                summary_data = []
                for window in selected_windows:
                    latest_data = {}
                    latest_data['Window'] = f"{window} days"
                    
                    for metric in ['return', 'sharpe', 'volatility', 'max_dd', 'win_rate']:
                        col_name = f'{metric}_{window}d'
                        if col_name in rolling_metrics.columns:
                            latest_value = rolling_metrics[col_name].iloc[-1]
                            
                            if metric in ['return', 'volatility', 'max_dd', 'win_rate']:
                                latest_data[metric.replace('_', ' ').title()] = f"{latest_value:.2%}"
                            else:
                                latest_data[metric.replace('_', ' ').title()] = f"{latest_value:.2f}"
                    
                    summary_data.append(latest_data)
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, hide_index=True, use_container_width=True)
            else:
                st.warning("Please select at least one time window to display.")
        else:
            st.info("Not enough data for rolling metrics calculation (minimum 30 days required)")
    
    with tab4:
        st.header("Trade History")
        
        # Display trading data with formatting
        if not trading_data.empty:
            display_data = trading_data.copy()
            display_data['realized_pnl'] = display_data['realized_pnl'].apply(lambda x: f"${x:,.2f}")
            display_data['opening_balance'] = display_data['opening_balance'].apply(lambda x: f"${x:,.2f}")
            display_data['closing_balance'] = display_data['closing_balance'].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(
                display_data[['date', 'opening_balance', 'closing_balance', 'realized_pnl', 'notes']],
                use_container_width=True,
                hide_index=True
            )
            
            # Export functionality
            csv = trading_data.to_csv(index=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"trading_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No trading data to display")

if __name__ == "__main__":
    main()
