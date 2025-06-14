# Stock Price Predictor

A web application that predicts stock prices using machine learning and displays financial metrics.

## Features

- Real-time stock data fetching using yfinance
- Machine learning-based price predictions
- Interactive stock charts
- Financial metrics display
- Technical indicators analysis

## Tech Stack

- Backend:
  - Python
  - Streamlit
  - Pandas
  - NumPy
  - Scikit-learn
  - yfinance

- Frontend:
  - React
  - TypeScript
  - Vite
  - Victory Charts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-price-predictor.git
cd stock-price-predictor
```

2. Install Python dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
```

## Usage

1. Start the backend server:
```bash
python -m streamlit run app.py
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

3. Open your browser and navigate to `http://localhost:5173`

## License

MIT 