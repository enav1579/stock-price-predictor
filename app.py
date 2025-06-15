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
    """Get historical stock data using yfinance"""
    try:
        # Calculate date range (20 years ago from today)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=20*365)  # 20 years
        
        # Get stock data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            st.error(f"No data found for {ticker}. Please check the symbol or try again later.")
            return None
            
        st.info(f"Showing 20 years of historical data from {start_date.strftime('%Y-%m-%d')}")
        return data
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        print(f"[DEBUG] Exception for {ticker}: {str(e)}")
        return None

def get_financial_data(ticker):
    """Fetch and combine financial statements"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=20*365)  # 20 years
        
        # Fetch data with explicit date range
        data = stock.history(start=start_date.strftime('%Y-%m-%d'), 
                           end=end_date.strftime('%Y-%m-%d'))
        
        if data.empty:
            st.error(f"No historical data found for {ticker}. Please check the symbol or try again later.")
            return None
            
        # Ensure the index is datetime
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
        
        st.info(f"Showing 20 years of historical data from {start_date.strftime('%Y-%m-%d')}")
        return data
        
    except Exception as e:
        st.error(f"Error fetching financial data: {str(e)}")
        return None

def calculate_indicators(data):
    """Calculate technical indicators for the stock data"""
    try:
        # Price-based features
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close']/data['Close'].shift(1))
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        # Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        
        # Price to MA ratios
        data['Price_to_SMA20'] = data['Close'] / data['SMA_20']
        data['Price_to_SMA50'] = data['Close'] / data['SMA_50']
        data['Price_to_SMA200'] = data['Close'] / data['SMA_200']
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
        
        # Volume indicators
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        data['Volume_Change'] = data['Volume'].pct_change()
        
        # Price momentum
        data['Momentum'] = data['Close'] - data['Close'].shift(10)
        data['Rate_of_Change'] = data['Close'].pct_change(periods=10)
        
        return data
        
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return data

def prepare_data(data):
    """Prepare data for model training"""
    try:
        if data is None or data.empty:
            st.error("No data available for preparation")
            return None, None, None
            
        # Calculate technical indicators
        data = calculate_indicators(data)
        
        # Create target variable (next day's closing price)
        data['Target'] = data['Close'].shift(-1)
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Select features for training
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'Log_Returns', 'Volatility',
            'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26',
            'Price_to_SMA20', 'Price_to_SMA50', 'Price_to_SMA200',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Middle', 'BB_Upper', 'BB_Lower',
            'Volume_MA', 'Volume_Ratio', 'Volume_Change',
            'Momentum', 'Rate_of_Change'
        ]
        
        X = data[features]
        y = data['Target']
        
        return X, y, data
        
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return None, None, None

def train_model(X, y):
    """Train the model with improved parameters"""
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Create and train the model with optimized parameters
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
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
        
        return model, train_rmse, test_rmse, train_r2, test_r2
        
    except Exception as e:
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
            X, y, data = prepare_data(data)
            if X is None or y is None:
                st.error("Error preparing data for analysis")
                return
                
            # Train model
            model, train_rmse, test_rmse, train_r2, test_r2 = train_model(X, y)
            if model is None:
                st.error("Error training prediction model")
                return
                
            # Make prediction
            last_data = X.iloc[-1:].copy()
            prediction = model.predict(last_data)[0]
            current_price = data['Close'].iloc[-1]
            predicted_price = current_price * (1 + prediction)
            
            # Display prediction
            st.markdown('<div class="prediction-header"><h2>Price Prediction</h2></div>', unsafe_allow_html=True)
            
            # Display prediction date
            next_trading_day = get_next_trading_day(data.index[-1])
            st.markdown(f'<div class="info-box">Prediction is for {next_trading_day.strftime("%Y-%m-%d")}</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Predicted Price", f"${predicted_price:.2f}")
            with col3:
                st.metric("Predicted Change", f"{prediction:.2%}")
            
            # Model Performance Analysis
            st.markdown('<div class="prediction-header"><h2>Model Performance Analysis</h2></div>', unsafe_allow_html=True)
            
            # Display model fit metrics
            st.markdown("### Model Fit Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training R² Score", f"{train_r2:.2%}")
            with col2:
                st.metric("Test R² Score", f"{test_r2:.2%}")
            
            # Display prediction accuracy metrics
            st.markdown("### Prediction Accuracy (Last 30 Days)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Absolute Error", f"${train_rmse:.2f}")
            with col2:
                st.metric("Mean Absolute Percentage Error", f"{test_rmse:.2f}%")
            
            # Plot actual vs predicted prices
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data.index[-30:],
                y=data['Close'][-30:],
                name='Actual Price',
                line=dict(color='#1E88E5', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index[-30:],
                y=model.predict(X[-30:]),
                name='Predicted Price',
                line=dict(color='#FFA726', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='Actual vs Predicted Prices (Last 30 Days)',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature Importance
            st.markdown('<div class="prediction-header"><h2>Feature Importance</h2></div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance in Price Prediction',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                template='plotly_white',
                showlegend=False,
                xaxis_title='Importance Score',
                yaxis_title='Feature'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display financial metrics
            st.markdown('<div class="prediction-header"><h2>Financial Analysis</h2></div>', unsafe_allow_html=True)
            
            # Create tabs for financial statements
            tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
            
            with tab1:
                display_financial_metrics(ticker, selected_years * 4, statement_type="income")
            with tab2:
                display_financial_metrics(ticker, selected_years * 4, statement_type="balance")
            with tab3:
                display_financial_metrics(ticker, selected_years * 4, statement_type="cash")
            
            # Add attribution
            add_attribution()
            
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please try again or contact support if the issue persists.")

if __name__ == "__main__":
    main() 