import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import time
from functools import wraps
from sklearn.preprocessing import MinMaxScaler
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
        padding: 2rem;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-header {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2196f3;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .attribution {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index (RSI)"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        st.error(f"Error calculating RSI: {str(e)}")
        return pd.Series(index=prices.index)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence (MACD)"""
    try:
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    except Exception as e:
        st.error(f"Error calculating MACD: {str(e)}")
        return pd.Series(index=prices.index), pd.Series(index=prices.index)

def get_stock_data(ticker):
    """Fetch stock data for the given ticker."""
    try:
        logging.debug(f"Fetching data for ticker: {ticker}")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)  # 5 years of data
        logging.debug(f"Date range: {start_date} to {end_date}")
        
        # Fetch data using yfinance
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            logging.error(f"No data available for ticker {ticker}")
            return None
            
        logging.debug(f"Data shape: {data.shape}")
        logging.debug(f"Data columns: {data.columns.tolist()}")
        logging.debug(f"Data index type: {type(data.index)}")
        
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
                logging.debug("Successfully converted index to datetime")
            except Exception as e:
                logging.error(f"Error converting index to datetime: {str(e)}")
                return None
        
        # Sort by date
        data = data.sort_index()
        logging.debug(f"Data after sorting:\n{data.head()}")
        
        return data
        
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def get_financial_data(ticker):
    """Fetch and combine financial statements"""
    try:
        logger.debug(f"Fetching data for ticker: {ticker}")
        stock = yf.Ticker(ticker)
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=20*365)  # 20 years
        
        logger.debug(f"Date range: {start_date} to {end_date}")
        
        # Fetch data with explicit date range
        data = stock.history(start=start_date.strftime('%Y-%m-%d'), 
                           end=end_date.strftime('%Y-%m-%d'))
        
        logger.debug(f"Data shape: {data.shape}")
        logger.debug(f"Data columns: {data.columns.tolist()}")
        logger.debug(f"Data index type: {type(data.index)}")
        
        if data.empty:
            logger.error(f"No data found for {ticker}")
            st.error(f"No historical data found for {ticker}. Please check the symbol or try again later.")
            return None
            
        # Convert index to datetime with explicit format
        try:
            data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
            logger.debug("Successfully converted index to datetime")
        except Exception as e:
            logger.error(f"Error converting index to datetime: {str(e)}")
            st.error(f"Error processing dates: {str(e)}")
            return None
            
        data = data.sort_index()
        logger.debug(f"Data after sorting: {data.head()}")
        
        st.info(f"Showing 20 years of historical data from {start_date.strftime('%Y-%m-%d')}")
        return data
        
    except Exception as e:
        logger.error(f"Error in get_financial_data: {str(e)}", exc_info=True)
        st.error(f"Error fetching financial data: {str(e)}")
        return None

def calculate_indicators(data):
    """Calculate technical indicators for the stock data"""
    try:
        logger.debug("Starting indicator calculation")
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Price-based features
        logger.debug("Calculating price-based features")
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Moving Averages
        logger.debug("Calculating moving averages")
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # Price to MA ratios
        logger.debug("Calculating price to MA ratios")
        df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
        df['Price_to_SMA50'] = df['Close'] / df['SMA_50']
        df['Price_to_SMA200'] = df['Close'] / df['SMA_200']
        
        # RSI
        logger.debug("Calculating RSI")
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        logger.debug("Calculating MACD")
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        logger.debug("Calculating Bollinger Bands")
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        
        # Volume indicators
        logger.debug("Calculating volume indicators")
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Price momentum
        logger.debug("Calculating price momentum")
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df['Rate_of_Change'] = df['Close'].pct_change(periods=10)
        
        # Additional features
        logger.debug("Calculating additional features")
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Range_Pct'] = df['Price_Range'] / df['Close']
        
        logger.debug("Indicator calculation completed")
        return df
        
    except Exception as e:
        logger.error(f"Error in calculate_indicators: {str(e)}", exc_info=True)
        st.error(f"Error calculating indicators: {str(e)}")
        return data

def prepare_data(data):
    """Prepare data for model training"""
    try:
        logging.debug("Starting data preparation")
        
        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Calculate volatility (20-day rolling standard deviation of returns)
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        # Calculate moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Calculate EMAs
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        
        # Calculate price to MA ratios
        data['Price_to_SMA20'] = data['Close'] / data['SMA_20']
        data['Price_to_SMA50'] = data['Close'] / data['SMA_50']
        data['Price_to_SMA200'] = data['Close'] / data['SMA_200']
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        # Calculate Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()
        
        # Calculate volume indicators
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        data['Volume_Change'] = data['Volume'].pct_change()
        
        # Calculate price momentum
        data['Momentum'] = data['Close'] - data['Close'].shift(10)
        data['Rate_of_Change'] = data['Close'].pct_change(periods=10)
        
        # Calculate additional features
        data['High_Low_Ratio'] = data['High'] / data['Low']
        data['Close_Open_Ratio'] = data['Close'] / data['Open']
        data['Price_Range'] = data['High'] - data['Low']
        data['Price_Range_Pct'] = data['Price_Range'] / data['Open']
        
        # Drop NaN values
        data = data.dropna()
        logging.debug(f"Data shape after dropping NaN: {data.shape}")
        
        # Create target variable (next day's closing price)
        y = data['Close'].shift(-1)
        
        # Select features for model
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Log_Returns', 
                         'Volatility', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
                         'Price_to_SMA20', 'Price_to_SMA50', 'Price_to_SMA200', 'RSI',
                         'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                         'Volume_MA', 'Volume_Ratio', 'Volume_Change', 'Momentum', 'Rate_of_Change',
                         'High_Low_Ratio', 'Close_Open_Ratio', 'Price_Range', 'Price_Range_Pct']
        
        X = data[feature_columns]
        
        # Drop the last row since we don't have the next day's price
        X = X[:-1]
        y = y[:-1]
        
        logging.debug(f"Final data shapes - X: {X.shape}, y: {y.shape}")
        
        return X, y
        
    except Exception as e:
        logging.error(f"Error preparing data: {str(e)}")
        raise

def train_model(X, y):
    """Train the model"""
    try:
        logger.debug("Starting model training")
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Create and train the model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        logger.debug(f"Model metrics - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
        logger.debug(f"Model metrics - Train R²: {train_r2:.2f}, Test R²: {test_r2:.2f}")
        
        return model, train_rmse, test_rmse, train_r2, test_r2
        
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}", exc_info=True)
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None

def display_financial_metrics(ticker, quarters, statement_type="income"):
    """Display financial metrics with improved error handling"""
    try:
        # Get financial data
        financial_data = get_financial_data(ticker)
        if financial_data is None or financial_data.empty:
            st.warning("Financial data not available for this ticker")
            return
            
        # Convert column names to datetime
        financial_data.columns = pd.to_datetime(financial_data.columns)
        
        # Sort columns by date in descending order
        financial_data = financial_data.sort_index(axis=1, ascending=False)
        
        # Get available dates
        available_dates = financial_data.columns
        if len(available_dates) < 2:
            st.warning(f"Only one quarter of data available for {ticker}. Historical comparison not possible.")
            return
            
        # Get the most recent date
        most_recent_date = available_dates[0]
        
        # Calculate the date for the specified number of quarters ago
        quarters_ago = most_recent_date - pd.DateOffset(months=3*quarters)
        
        # Filter columns to include only the quarters in the selected period
        selected_dates = [col for col in available_dates if col >= quarters_ago]
        
        # If we don't have enough data for the requested quarters, use what we have
        if len(selected_dates) < 2:
            actual_quarters = len(available_dates)
            st.info(f"Only {actual_quarters} quarters of data available. Showing comparison for available period.")
            selected_dates = available_dates[:2]  # Use the two most recent quarters
        else:
            # Use all available quarters in the selected period
            selected_dates = sorted(selected_dates, reverse=True)  # Ensure descending order
            
        financial_data = financial_data[selected_dates]
        
        # Define metrics based on statement type
        if statement_type == "income":
            metrics = {
                'Total Revenue': 'Revenue',
                'Cost Of Revenue': 'Cost of Revenue',
                'Gross Profit': 'Gross Profit',
                'Research Development': 'R&D Expenses',
                'Selling General And Administrative': 'SG&A Expenses',
                'Operating Income': 'Operating Income',
                'Interest Expense': 'Interest Expense',
                'Income Before Tax': 'Income Before Tax',
                'Income Tax Expense': 'Income Tax Expense',
                'Net Income': 'Net Income',
                'Basic EPS': 'Basic EPS',
                'Diluted EPS': 'Diluted EPS'
            }
        elif statement_type == "balance":
            metrics = {
                'Total Assets': 'Total Assets',
                'Total Current Assets': 'Current Assets',
                'Cash': 'Cash & Equivalents',
                'Short Term Investments': 'Short Term Investments',
                'Net Receivables': 'Accounts Receivable',
                'Inventory': 'Inventory',
                'Total Liabilities': 'Total Liabilities',
                'Total Current Liabilities': 'Current Liabilities',
                'Short Long Term Debt': 'Short Term Debt',
                'Long Term Debt': 'Long Term Debt',
                'Total Stockholder Equity': 'Shareholders Equity',
                'Retained Earnings': 'Retained Earnings'
            }
        else:  # cash flow
            metrics = {
                'Operating Cash Flow': 'Operating Cash Flow',
                'Investing Cash Flow': 'Investing Cash Flow',
                'Financing Cash Flow': 'Financing Cash Flow',
                'Free Cash Flow': 'Free Cash Flow',
                'Capital Expenditure': 'Capital Expenditure',
                'Dividends Paid': 'Dividends Paid',
                'Net Income': 'Net Income',
                'Depreciation': 'Depreciation',
                'Change In Cash': 'Change in Cash',
                'Change In Inventory': 'Change in Inventory',
                'Change In Receivables': 'Change in Receivables',
                'Change In Payables': 'Change in Payables'
            }
        
        # Create a DataFrame for display
        display_data = []
        for metric, display_name in metrics.items():
            if metric in financial_data.index:
                row_data = {'Metric': display_name}
                # Add each quarter's data
                for date in selected_dates:
                    value = financial_data.loc[metric, date]
                    row_data[date.strftime('%Y-%m-%d')] = value
                display_data.append(row_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(display_data)
        
        # Format the DataFrame
        def format_currency(x):
            if pd.isna(x):
                return ''
            return f"${x:,.0f}"
        
        # Create styled DataFrame
        def style_cell(x, df):
            if x.name == 'Metric':
                return ['background-color: #f8f9fa' for _ in x]
            
            # Get the column index
            col_idx = df.columns.get_loc(x.name)
            if col_idx == 0:  # Skip the Metric column
                return [''] * len(x)
            
            # Get the previous quarter's values
            prev_col = df.columns[col_idx - 1]
            prev_values = df[prev_col]
            
            # Compare with previous quarter
            styles = []
            for val, prev_val in zip(x, prev_values):
                try:
                    # Convert to float for comparison
                    val_float = float(val) if not pd.isna(val) else None
                    prev_float = float(prev_val) if not pd.isna(prev_val) else None
                    
                    if val_float is None or prev_float is None:
                        styles.append('')
                    else:
                        # For certain metrics, higher values are better (green)
                        # For others, lower values are better (red)
                        if statement_type == "income":
                            # For income statement, higher values are generally better
                            color = 'color: #28a745' if val_float > prev_float else 'color: #dc3545' if val_float < prev_float else ''
                        elif statement_type == "balance":
                            # For balance sheet, it depends on the metric
                            if x.name in ['Total Liabilities', 'Total Current Liabilities', 'Short Long Term Debt', 'Long Term Debt']:
                                # For liabilities, lower values are better
                                color = 'color: #dc3545' if val_float > prev_float else 'color: #28a745' if val_float < prev_float else ''
                            else:
                                # For assets and equity, higher values are better
                                color = 'color: #28a745' if val_float > prev_float else 'color: #dc3545' if val_float < prev_float else ''
                        else:  # cash flow
                            # For cash flow, it depends on the metric
                            if x.name in ['Investing Cash Flow', 'Capital Expenditure', 'Dividends Paid']:
                                # For these metrics, lower values are better
                                color = 'color: #dc3545' if val_float > prev_float else 'color: #28a745' if val_float < prev_float else ''
                            else:
                                # For other cash flow metrics, higher values are better
                                color = 'color: #28a745' if val_float > prev_float else 'color: #dc3545' if val_float < prev_float else ''
                        styles.append(color)
                except (ValueError, TypeError):
                    styles.append('')
            
            return styles
        
        styled_df = df.style.format({
            col: format_currency for col in df.columns if col != 'Metric'
        }).apply(style_cell, df=df)
        
        # Display the DataFrame
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )
        
    except Exception as e:
        st.error(f"Error fetching financial data: {str(e)}")
        st.info("Please check if the ticker symbol is correct and try again.")

def safe_pct_change(data):
    """Calculate percentage changes safely"""
    try:
        # Convert to float and handle any non-numeric values
        numeric_data = data.astype(float)
        
        # Calculate percentage changes without filling NA values
        pct_changes = numeric_data.pct_change(axis=1, fill_method=None) * 100
        
        # Replace infinite values with NaN
        pct_changes = pct_changes.replace([np.inf, -np.inf], np.nan)
        
        return pct_changes
        
    except Exception as e:
        st.error(f"Error calculating percentage changes: {str(e)}")
        return pd.DataFrame()

def add_attribution():
    """Add attribution footer"""
    st.markdown("""
    <div class="attribution">
        <p>Created by Charles Eric Navarro with Cursor</p>
        <p>Version 1.0 ({})</p>
        <p>© 2024 All rights reserved. This program cannot be copied or modified without express consent from the creator.</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)

def get_next_trading_day(last_date):
    """Calculate the next trading day"""
    next_day = last_date + timedelta(days=1)
    while next_day.weekday() >= 5:  # Skip weekends (5=Saturday, 6=Sunday)
        next_day += timedelta(days=1)
    return next_day

def generate_future_dates(last_date, days=252):
    """Generate future trading dates for the next year"""
    try:
        future_dates = []
        current_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)
        while len(future_dates) < days:
            if current_date.weekday() < 5:  # Only weekdays
                future_dates.append(current_date)
            current_date += pd.Timedelta(days=1)
        return pd.DatetimeIndex(future_dates)
    except Exception as e:
        logger.error(f"Error generating future dates: {str(e)}", exc_info=True)
        return None

def predict_future_prices(model, last_data, features, days=252):
    """Predict future prices for the next year"""
    try:
        future_dates = generate_future_dates(last_data.index[-1], days)
        if future_dates is None:
            return None
            
        predictions = []
        current_data = last_data.copy()
        
        for date in future_dates:
            # Prepare features for prediction
            X_pred = current_data[features].iloc[-1:].copy()
            
            # Make prediction
            pred_price = model.predict(X_pred)[0]
            predictions.append(pred_price)
            
            # Update current data for next prediction
            new_row = current_data.iloc[-1:].copy()
            new_row.index = [date]
            new_row['Close'] = pred_price
            new_row['Open'] = pred_price
            new_row['High'] = pred_price
            new_row['Low'] = pred_price
            
            # Update features
            new_row['Returns'] = (pred_price / current_data['Close'].iloc[-1]) - 1
            new_row['Log_Returns'] = np.log(pred_price / current_data['Close'].iloc[-1])
            new_row['SMA_20'] = current_data['Close'].iloc[-19:].mean()
            new_row['SMA_50'] = current_data['Close'].iloc[-49:].mean()
            new_row['SMA_200'] = current_data['Close'].iloc[-199:].mean()
            new_row['Volume'] = current_data['Volume'].iloc[-20:].mean()
            new_row['Volume_MA'] = current_data['Volume'].iloc[-20:].mean()
            new_row['Volume_Ratio'] = 1.0
            new_row['Momentum'] = pred_price - current_data['Close'].iloc[-10]
            new_row['Volatility'] = current_data['Returns'].iloc[-20:].std()
            
            current_data = pd.concat([current_data, new_row])
        
        return pd.Series(predictions, index=future_dates)
        
    except Exception as e:
        logger.error(f"Error in predict_future_prices: {str(e)}", exc_info=True)
        st.error(f"Error predicting future prices: {str(e)}")
        return None

def plot_predictions(historical_data, predictions):
    """Create an interactive plot of historical and predicted prices"""
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'],
        name='Historical Prices',
        line=dict(color='blue')
    ))
    
    # Plot predictions
    fig.add_trace(go.Scatter(
        x=predictions.index,
        y=predictions,
        name='Predicted Prices',
        line=dict(color='red', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def main():
    """Main function with improved error handling"""
    try:
        # Sidebar
        with st.sidebar:
            st.markdown('<div class="sidebar-header"><h1>Stock Price Predictor</h1></div>', unsafe_allow_html=True)
            
            # Ticker input
            ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
            
            # Financial statement period selection
            years = list(range(1, 6))  # 1 to 5 years
            selected_years = st.selectbox(
                "Select Financial Statement Period",
                options=years,
                index=3,  # Default to 4 years
                format_func=lambda x: f"{x} {'Year' if x == 1 else 'Years'}"
            )
            
            # How it works section
            st.markdown("### How It Works")
            st.markdown("""
            This application uses machine learning to predict stock prices based on:
            
            📊 **Technical Indicators:**
            - RSI (Relative Strength Index)
            - MACD (Moving Average Convergence Divergence)
            - Moving Averages
            
            📈 **Price Data:**
            - Historical prices
            - Volume
            - Price momentum
            
            💰 **Financial Analysis:**
            - Income Statement
            - Balance Sheet
            - Cash Flow Statement
            
            The model is trained on up to 20 years of historical data and provides:
            - Next day price prediction
            - Model performance metrics
            - Feature importance analysis
            - Financial statement analysis
            """)
        
        # Main content
        st.markdown("# Stock Price Predictor")
        
        if st.button("Predict"):
            # Get stock data
            data = get_stock_data(ticker)
            if data is None:
                st.error("Please enter a valid ticker symbol")
                return
                
            # Prepare data
            X, y = prepare_data(data)
            if X is None or y is None:
                st.error("Error preparing data for analysis")
                return
                
            # Train model
            model, train_rmse, test_rmse, train_r2, test_r2 = train_model(X, y)
            if model is None:
                st.error("Error training prediction model")
                return
                
            # Make predictions
            features = X.columns.tolist()
            predictions = predict_future_prices(model, data, features)
            
            if predictions is not None:
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training RMSE", f"${train_rmse:.2f}")
                    st.metric("Training R²", f"{train_r2:.2f}")
                with col2:
                    st.metric("Testing RMSE", f"${test_rmse:.2f}")
                    st.metric("Testing R²", f"{test_r2:.2f}")
                
                # Plot predictions
                fig = plot_predictions(data, predictions)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display prediction details
                st.subheader("Prediction Details")
                last_price = data['Close'].iloc[-1]
                predicted_price = predictions.iloc[-1]
                price_change = predicted_price - last_price
                percent_change = (price_change / last_price) * 100
                
                st.metric(
                    "Predicted Price in 1 Year",
                    f"${predicted_price:.2f}",
                    f"{percent_change:+.2f}%"
                )
                
                # Display prediction table
                st.subheader("Monthly Predictions")
                monthly_predictions = predictions.resample('M').last()
                monthly_predictions = pd.DataFrame({
                    'Predicted Price': monthly_predictions,
                    'Month': monthly_predictions.index.strftime('%B %Y')
                })
                st.dataframe(monthly_predictions)
                
                # Display financial metrics
                display_financial_metrics(ticker, selected_years * 4, statement_type="income")
                
                # Add attribution
                add_attribution()
                
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please try again or contact support if the issue persists.")

if __name__ == "__main__":
    main() 