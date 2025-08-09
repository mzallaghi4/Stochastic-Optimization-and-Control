## Downloads real market data using yfinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import yfinance as yf

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define assets and download historical data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start_date = '2020-01-01'
end_date = '2025-01-01'

data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
print("Stock data downloaded successfully!")
