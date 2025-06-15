import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Add a title
st.title("Stock Price Predictor 📈")

# Add a description
st.write("""
This app shows stock price data and basic statistics. Enter a stock ticker symbol to get started.
""")

# Add a text input for the stock ticker
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, GOOGL):", "AAPL").upper()

# Add a button to trigger the data fetch
if st.button("Show Stock Data"):
    try:
        # Show a loading spinner
        with st.spinner("Fetching stock data..."):
            # Get stock data
            stock = yf.Ticker(ticker)
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # Get 1 year of data
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                st.error(f"No data found for {ticker}. Please check the symbol and try again.")
            else:
                # Display the data
                st.write(f"### Historical Data for {ticker}")
                st.dataframe(data.tail())
                
                # Create a line chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#17BECF')
                ))
                
                fig.update_layout(
                    title=f"{ticker} Stock Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show some basic statistics
                st.write("### Basic Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"${data['Close'][-1]:.2f}")
                with col2:
                    st.metric("52-Week High", f"${data['High'].max():.2f}")
                with col3:
                    st.metric("52-Week Low", f"${data['Low'].min():.2f}")
                
                # Add volume chart
                st.write("### Trading Volume")
                volume_fig = go.Figure()
                volume_fig.add_trace(go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='#17BECF'
                ))
                
                volume_fig.update_layout(
                    title=f"{ticker} Trading Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    template="plotly_white"
                )
                
                st.plotly_chart(volume_fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try again with a different ticker symbol.")

# Add a footer
st.write("---")
st.write("Made with ❤️ using Streamlit") 