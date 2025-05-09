import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# --- Configuration ---
RISK_FREE_RATE = 0.0425  # Approximate 3-Month T-bill rate (annualized) as of May 2025
TRADING_DAYS_PER_YEAR = 365 # Crypto trades 24/7

# --- Data Fetching (with Streamlit caching) ---
@st.cache_data(ttl=6*60*60) # Cache data for 6 hours
def get_btc_data(years=5):
    """
    Fetches historical BTC-USD data for the last 'years' years.
    Tries 'Adj Close' first, then 'Close'. Ensures a 1D Series is returned for prices.
    """
    _end_date = datetime.now()
    _start_date = _end_date - timedelta(days=years * 365 + (years // 4)) # Account for leap years approx.
    
    st.info(f"Fetching BTC-USD data from {_start_date.strftime('%Y-%m-%d')} to {_end_date.strftime('%Y-%m-%d')}...")
    try:
        btc_data_full = yf.download('BTC-USD', start=_start_date.strftime('%Y-%m-%d'), end=_end_date.strftime('%Y-%m-%d'))
        
        if btc_data_full.empty:
            st.error("No data fetched from yfinance. The DataFrame is empty.")
            return None, None, None
        
        # For debugging, you can uncomment this to see the structure of the fetched columns:
        # st.write("Available columns in fetched data:", btc_data_full.columns)
        # if isinstance(btc_data_full.columns, pd.MultiIndex):
        #     st.write("Columns are MultiIndex.")

        price_column_to_use = None
        if 'Adj Close' in btc_data_full.columns:
            price_column_to_use = 'Adj Close'
        elif 'Close' in btc_data_full.columns:
            price_column_to_use = 'Close'
            st.info("Using 'Close' price column as 'Adj Close' was not found (this is common for crypto).")
        else:
            st.error(f"Neither 'Adj Close' nor 'Close' columns were found in the fetched data. Available columns: {btc_data_full.columns.tolist()}")
            return None, None, None
            
        # Extract the potential price data
        price_data = btc_data_full[price_column_to_use]
        
        # Ensure price_data is a 1D Series
        if isinstance(price_data, pd.DataFrame):
            # If yfinance returned a DataFrame (e.g., due to MultiIndex columns where 'price_column_to_use' was a top level)
            if price_data.shape[1] == 1:
                # If it's a DataFrame with exactly one column, squeeze it into a Series
                price_data = price_data.iloc[:, 0]
                # st.info(f"Price data for '{price_column_to_use}' was a DataFrame; converted to Series.")
            else:
                # This would be unusual for a single price type of a single ticker
                st.error(f"Extracted price data for '{price_column_to_use}' is a DataFrame with multiple columns: {price_data.columns.tolist()}. Cannot proceed.")
                return None, None, None
        
        # Final check if it's a Series
        if not isinstance(price_data, pd.Series):
            st.error(f"Extracted price data for '{price_column_to_use}' is not a Pandas Series after processing. Type: {type(price_data)}")
            return None, None, None

        # Ensure the Series has a name, useful for plotting or other operations
        if price_data.name is None and price_column_to_use is not None:
             price_data.name = price_column_to_use
            
        st.success(f"Data fetched successfully. Using '{price_column_to_use}' prices.")
        return price_data, btc_data_full.index.min(), btc_data_full.index.max()
        
    except Exception as e:
        st.error(f"An error occurred during data fetching or processing: {e}")
        # For more detailed debugging in your local environment or Streamlit Cloud logs:
        # import traceback
        # st.error(traceback.format_exc())
        return None, None, None


# --- Metric Calculation Functions (same as before) ---
def calculate_daily_returns(series):
    return series.pct_change().dropna()

def calculate_cagr(series, years):
    if len(series) < 2 or years <= 0: return 0.0
    start_value = series.iloc[0]
    end_value = series.iloc[-1]
    if start_value == 0: return 0.0
    cagr = (end_value / start_value) ** (1 / years) - 1
    return cagr

def calculate_annualized_volatility(daily_returns):
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

def calculate_max_drawdown(series):
    roll_max = series.cummax()
    daily_drawdown = series / roll_max - 1.0
    max_drawdown = daily_drawdown.cummin().iloc[-1] if not daily_drawdown.empty else 0.0
    return max_drawdown

def calculate_sharpe_ratio(daily_returns, risk_free_rate_annual):
    if daily_returns.empty or daily_returns.std() == 0: return np.nan
    daily_risk_free_rate = (1 + risk_free_rate_annual)**(1/TRADING_DAYS_PER_YEAR) - 1
    excess_returns = daily_returns - daily_risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return sharpe_ratio

def calculate_sortino_ratio(daily_returns, risk_free_rate_annual):
    if daily_returns.empty: return np.nan
    daily_risk_free_rate = (1 + risk_free_rate_annual)**(1/TRADING_DAYS_PER_YEAR) - 1
    excess_returns = daily_returns - daily_risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.empty: return np.nan
    downside_deviation = downside_returns.std()
    if downside_deviation == 0: return np.nan if excess_returns.mean() <=0 else np.inf
    sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(TRADING_DAYS_PER_YEAR)
    return sortino_ratio

def calculate_calmar_ratio(cagr, max_drawdown):
    if max_drawdown == 0: return np.nan
    return cagr / abs(max_drawdown)

# --- Streamlit App Layout ---
st.set_page_config(page_title="BTC Performance Analyzer", layout="wide")
st.title("📈 Bitcoin Performance Analyzer")
st.markdown(f"Analyze Bitcoin (BTC-USD) performance over a selected period (max 5 years). Current Risk-Free Rate used for calculations: **{RISK_FREE_RATE*100:.2f}%** (annualized).")
st.caption(f"Current date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Data is cached for 6 hours)")

# 1. Get all data (last 5 years)
max_history_years = 5
all_btc_prices, min_date_available, max_date_available = get_btc_data(years=max_history_years)

if all_btc_prices is not None and not all_btc_prices.empty:
    st.sidebar.header("🗓️ Select Date Range")
    
    # Ensure min_date_available and max_date_available are datetime objects
    if isinstance(min_date_available, pd.Timestamp):
        min_date_available = min_date_available.to_pydatetime().date()
    if isinstance(max_date_available, pd.Timestamp):
        max_date_available = max_date_available.to_pydatetime().date()

    # Default start date: 1 year ago or min_date_available if less than 1 year data
    default_start_date = max(min_date_available, max_date_available - timedelta(days=365))
    
    user_start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start_date,
        min_value=min_date_available,
        max_value=max_date_available - timedelta(days=1) # Ensure end date can be at least one day after start
    )
    
    user_end_date = st.sidebar.date_input(
        "End Date",
        value=max_date_available,
        min_value=user_start_date + timedelta(days=1), # Ensure end date is after start date
        max_value=max_date_available
    )

    # Max 5 year window check from the user's chosen start date
    if (user_end_date - user_start_date).days > max_history_years * 365.25 + 1 : # Added .25 for leap year avg
        st.sidebar.warning(f"Selected range exceeds {max_history_years} years. Analysis will be limited or consider adjusting dates.")
        # You could automatically adjust one of the dates if you prefer:
        # user_end_date = user_start_date + timedelta(days=max_history_years * 365)
        # if user_end_date > max_date_available: user_end_date = max_date_available

    # Filter data for the selected range (ensure user_start_date and user_end_date are pandas-compatible)
    # Convert date objects to datetime64[ns] for pandas indexing
    user_start_datetime = pd.to_datetime(user_start_date)
    user_end_datetime = pd.to_datetime(user_end_date)

    selected_prices = all_btc_prices[(all_btc_prices.index >= user_start_datetime) & 
                                     (all_btc_prices.index <= user_end_datetime)]

    if selected_prices.empty or len(selected_prices) < 2:
        st.warning("Not enough data for the selected range to perform calculations. Please select a wider range.")
    else:
        st.header(f"Performance: {user_start_date.strftime('%Y-%m-%d')} to {user_end_date.strftime('%Y-%m-%d')}")

        # Calculate metrics
        daily_returns = calculate_daily_returns(selected_prices)
        
        if daily_returns.empty:
            st.warning("Could not calculate daily returns. Need at least two data points in the selected range.")
        else:
            num_days = (selected_prices.index.max() - selected_prices.index.min()).days
            years_in_period = num_days / TRADING_DAYS_PER_YEAR if num_days > 0 else 0

            cagr = calculate_cagr(selected_prices, years_in_period)
            ann_volatility = calculate_annualized_volatility(daily_returns)
            max_dd = calculate_max_drawdown(selected_prices)
            sharpe = calculate_sharpe_ratio(daily_returns, RISK_FREE_RATE)
            sortino = calculate_sortino_ratio(daily_returns, RISK_FREE_RATE)
            calmar = calculate_calmar_ratio(cagr, max_dd)
            total_return = (selected_prices.iloc[-1] / selected_prices.iloc[0]) - 1 if len(selected_prices) > 0 and selected_prices.iloc[0] != 0 else 0

            # Display Metrics
            st.subheader("Key Performance Indicators (KPIs)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Return", f"{total_return*100:.2f}%", f"${selected_prices.iloc[-1]:,.2f} (End) / ${selected_prices.iloc[0]:,.2f} (Start)")
            col2.metric("Annualized Return (CAGR)", f"{cagr*100:.2f}%")
            col3.metric("Annualized Volatility", f"{ann_volatility*100:.2f}%")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Max Drawdown", f"{max_dd*100:.2f}%")
            col5.metric("Sharpe Ratio", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")
            col6.metric("Sortino Ratio", f"{sortino:.2f}" if not np.isnan(sortino) else "N/A")

            st.metric("Calmar Ratio", f"{calmar:.2f}" if not np.isnan(calmar) else "N/A")
            
            st.markdown(f"**Period:** {selected_prices.index.min().strftime('%Y-%m-%d')} to {selected_prices.index.max().strftime('%Y-%m-%d')} ({num_days} days)")


            # Plotting
            st.subheader("Price Chart")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(selected_prices.index, selected_prices.values, label='BTC Price (USD)', color='orange')
            ax.set_title(f'BTC-USD Price Performance')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD)')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display Data Table (optional)
            if st.checkbox("Show Price Data Table for Selected Range"):
                st.dataframe(selected_prices.to_frame(name="Price").sort_index(ascending=False))

else:
    st.error("Failed to load Bitcoin price data. The app cannot continue.")

st.sidebar.markdown("---")
st.sidebar.info("This app shows the performance of holding Bitcoin (BTC). Adjust the date range to see metrics for different periods.")
st.sidebar.markdown("Powered by [Streamlit](https://streamlit.io) & [yfinance](https://pypi.org/project/yfinance/)")
