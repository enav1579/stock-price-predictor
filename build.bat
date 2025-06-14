@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt

echo Building executable...
pyinstaller --clean --onefile --name StockPricePredictor ^
  --hidden-import streamlit ^
  --hidden-import pandas ^
  --hidden-import yfinance ^
  --hidden-import plotly ^
  --hidden-import sklearn ^
  --hidden-import numpy ^
  --hidden-import peewee ^
  app.py

echo Build complete! The executable is in the dist folder.
pause 