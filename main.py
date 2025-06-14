from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from pydantic import BaseModel
from services.stock_data import get_stock_metrics, get_historical_data
from services.predictions import predict_next_day, get_technical_indicators

app = FastAPI(
    title="Stock Market API",
    description="API for fetching stock market data, metrics, and predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockMetrics(BaseModel):
    ticker: str
    price: Optional[float] = None
    change: Optional[float] = None
    pe: Optional[float] = None
    eps: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    market_cap: Optional[float] = None
    volume: Optional[int] = None
    error: Optional[str] = None

class HistoricalDataPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class HistoricalData(BaseModel):
    ticker: str
    data: Optional[List[HistoricalDataPoint]] = None
    error: Optional[str] = None

class PredictionResult(BaseModel):
    ticker: str
    current_price: Optional[float] = None
    predicted_price: Optional[float] = None
    predicted_change: Optional[float] = None
    confidence: Optional[float] = None
    features_importance: Optional[Dict[str, float]] = None
    error: Optional[str] = None

class TechnicalIndicators(BaseModel):
    ticker: str
    indicators: Optional[Dict[str, float]] = None
    error: Optional[str] = None

@app.get("/stocks/metrics", response_model=List[StockMetrics])
def stock_metrics(tickers: List[str] = Query(..., description="List of stock tickers")):
    """
    Get current metrics for multiple stock tickers.
    
    - **tickers**: List of stock ticker symbols (e.g., ["AAPL", "MSFT", "GOOGL"])
    """
    return get_stock_metrics(tickers)

@app.get("/stocks/historical/{ticker}", response_model=HistoricalData)
def historical_data(
    ticker: str,
    period: str = Query("1mo", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    interval: str = Query("1d", description="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)")
):
    """
    Get historical stock data for a specific ticker.
    
    - **ticker**: Stock ticker symbol
    - **period**: Time period to fetch
    - **interval**: Data interval
    """
    return get_historical_data(ticker, period, interval)

@app.get("/stocks/predict/{ticker}", response_model=PredictionResult)
def predict_stock(ticker: str):
    """
    Get price prediction for a specific ticker.
    
    - **ticker**: Stock ticker symbol
    """
    return predict_next_day(ticker)

@app.get("/stocks/indicators/{ticker}", response_model=TechnicalIndicators)
def technical_indicators(ticker: str):
    """
    Get technical indicators for a specific ticker.
    
    - **ticker**: Stock ticker symbol
    """
    return get_technical_indicators(ticker) 