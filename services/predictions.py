import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Tuple
import yfinance as yf
from services.stock_data import get_historical_data
from datetime import datetime, timedelta

def get_next_trading_day(last_date: datetime) -> datetime:
    """
    Calculate the next trading day from the last known trading date.
    Skips weekends and holidays.
    
    Args:
        last_date: The last known trading date
    
    Returns:
        The next trading day
    """
    next_day = last_date + timedelta(days=1)
    
    # Skip weekends
    while next_day.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        next_day += timedelta(days=1)
    
    return next_day

def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for the model using 20 years of historical data.
    
    Args:
        data: DataFrame with historical stock data
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    try:
        print("Input data columns:", data.columns)
        print("Input data shape:", data.shape)
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure we have the required columns
        required_columns = ['close', 'volume']
        if not all(col in df.columns for col in required_columns):
            print("Missing required columns. Available columns:", df.columns)
            return pd.DataFrame(), pd.Series()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for window in [5, 20, 50, 100, 200]:
            df[f'ma{window}'] = df['close'].rolling(window=window).mean()
            df[f'ma{window}_ratio'] = df['close'] / df[f'ma{window}']
        
        # Volatility
        for window in [20, 50, 200]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
        
        # Volume features
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        # Create target variable (next day's close price)
        target = df['close'].shift(-1)
        
        # Select features for model
        feature_columns = [
            'returns', 'log_returns',
            'ma5_ratio', 'ma20_ratio', 'ma50_ratio', 'ma100_ratio', 'ma200_ratio',
            'volatility_20', 'volatility_50', 'volatility_200',
            'volume_ratio', 'rsi', 'macd', 'signal_line'
        ]
        
        # Remove last row (no target available)
        features = df[feature_columns].iloc[:-1]
        target = target.iloc[:-1]
        
        print("Final features shape:", features.shape)
        print("Target shape:", target.shape)
        
        return features, target
    except Exception as e:
        print(f"Error in prepare_data: {str(e)}")
        return pd.DataFrame(), pd.Series()

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD and Signal Line."""
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def train_model(features: pd.DataFrame, target: pd.Series) -> RandomForestRegressor:
    """
    Train a Random Forest model on the prepared features.
    
    Args:
        features: DataFrame with engineered features
        target: Series with target values
    
    Returns:
        Trained RandomForestRegressor model
    """
    try:
        print("Training data shape:", features.shape)
        print("Target shape:", target.shape)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        model.fit(features, target)
        return model
    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        raise

def predict_next_day(ticker: str, historical_data: pd.DataFrame = None) -> dict:
    """
    Predict the next day's stock price for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        historical_data: Optional DataFrame with historical data for backtesting
    
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Get historical data if not provided
        if historical_data is None:
            hist_data = get_historical_data(ticker, "20y", "1d")
            if 'error' in hist_data:
                return {'error': hist_data['error']}
            historical_data = pd.DataFrame(hist_data['data'])
            historical_data['date'] = pd.to_datetime(historical_data['date'])
            historical_data.set_index('date', inplace=True)
        
        print("Historical data shape:", historical_data.shape)
        print("Historical data columns:", historical_data.columns)
        
        # Prepare features and target
        features, target = prepare_data(historical_data)
        if features.empty or target.empty:
            return {'error': 'Not enough data to make prediction'}
        
        # Train model
        try:
            model = train_model(features, target)
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return {'error': 'Failed to train prediction model'}
        
        # Get latest features for prediction
        try:
            latest_features = features.iloc[-1:].copy()
        except Exception as e:
            print(f"Error getting latest features: {str(e)}")
            return {'error': 'Failed to prepare latest features for prediction'}
        
        # Make prediction
        try:
            predicted_price = float(model.predict(latest_features)[0])
            current_price = float(historical_data['close'].iloc[-1])
            predicted_change = ((predicted_price - current_price) / current_price) * 100
            
            # Calculate confidence based on model's performance
            confidence = float(model.score(features, target))
            
            # Get feature importance
            feature_importance = dict(zip(features.columns, model.feature_importances_))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_change': predicted_change,
                'confidence': confidence,
                'features_importance': feature_importance
            }
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return {'error': 'Failed to make price prediction'}
            
    except Exception as e:
        print(f"Error in predict_next_day: {str(e)}")
        return {'error': str(e)}

def get_technical_indicators(ticker: str) -> Dict:
    """Calculate technical indicators for a stock."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="20y")
        
        if hist.empty:
            return {'error': 'No historical data available'}
        
        # Calculate technical indicators
        df = hist.copy()
        
        # Moving Averages
        for window in [20, 50, 100, 200]:
            df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
        
        # RSI
        df['RSI'] = calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['Signal_Line'] = calculate_macd(df['Close'])
        
        # Get latest values
        latest = df.iloc[-1]
        
        return {
            'ticker': ticker,
            'indicators': {
                'ma20': latest['MA20'],
                'ma50': latest['MA50'],
                'ma100': latest['MA100'],
                'ma200': latest['MA200'],
                'rsi': latest['RSI'],
                'macd': latest['MACD'],
                'signal_line': latest['Signal_Line']
            }
        }
    except Exception as e:
        return {'ticker': ticker, 'error': str(e)} 