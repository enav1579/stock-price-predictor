import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

def get_stock_metrics(tickers: List[str]) -> List[Dict]:
    results = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            data = {
                'ticker': ticker,
                'price': info.get('regularMarketPrice'),
                'change': info.get('regularMarketChangePercent'),
                'pe': info.get('trailingPE'),
                'eps': info.get('trailingEps'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'market_cap': info.get('marketCap'),
                'volume': info.get('regularMarketVolume'),
            }
            results.append(data)
        except Exception as e:
            results.append({'ticker': ticker, 'error': str(e)})
    return results

def get_historical_data(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d"
) -> Dict[str, Union[str, List[Dict]]]:
    """
    Get historical stock data for a given ticker.
    If 20 years of data isn't available, it will use the maximum available data.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        Dictionary containing historical data or error message
    """
    try:
        stock = yf.Ticker(ticker)
        
        # First try to get 20 years of data
        hist = stock.history(period="20y", interval=interval)
        
        # If we don't have enough data, try to get maximum available
        if len(hist) < 252:  # Less than 1 year of trading days
            hist = stock.history(period="max", interval=interval)
            
        if hist.empty:
            return {'ticker': ticker, 'error': 'No historical data available'}
        
        # Convert column names to lowercase
        hist.columns = [col.lower() for col in hist.columns]
        
        # Convert DataFrame to list of dictionaries
        historical_data = []
        for index, row in hist.iterrows():
            try:
                historical_data.append({
                    'date': index.strftime('%Y-%m-%d'),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume'])
                })
            except (ValueError, TypeError) as e:
                print(f"Error processing row {index}: {str(e)}")
                continue
        
        # Calculate the actual time period of the data
        if len(historical_data) > 0:
            start_date = historical_data[0]['date']
            end_date = historical_data[-1]['date']
            data_period = f"from {start_date} to {end_date}"
        else:
            data_period = "unknown period"
        
        return {
            'ticker': ticker,
            'data': historical_data,
            'period': data_period,
            'data_points': len(historical_data)
        }
    except Exception as e:
        print(f"Error in get_historical_data: {str(e)}")
        return {'ticker': ticker, 'error': str(e)} 