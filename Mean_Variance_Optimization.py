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
end_date = '2023-01-01'

data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
print("Data downloaded successfully!")


returns = data.pct_change().dropna()
print(f"\nReturns shape: {returns.shape}")
print(returns.head())

expected_returns = returns.mean() * 252  # Annualized
cov_matrix = returns.cov() * 252  # Annualized
print("\nExpected Annual Returns:")
print(expected_returns)
print("\nCovariance Matrix:")
print(cov_matrix)

def portfolio_performance(weights, expected_returns, cov_matrix):
    """Calculate portfolio return and volatility"""
    returns = np.sum(expected_returns * weights)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, volatility

def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.02):
    """Calculate negative Sharpe ratio (for minimization)"""
    returns, volatility = portfolio_performance(weights, expected_returns, cov_matrix)
    return -(returns - risk_free_rate) / volatility

def portfolio_volatility(weights, expected_returns, cov_matrix):
    """Calculate portfolio volatility"""
    return portfolio_performance(weights, expected_returns, cov_matrix)[1]

def portfolio_return(weights, expected_returns, cov_matrix):
    """Calculate portfolio return"""
    return portfolio_performance(weights, expected_returns, cov_matrix)[0]

num_assets = len(tickers)
constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
bounds = tuple((0, 1) for _ in range(num_assets))  # No short selling

# Initial guess (equal weights)
initial_weights = np.array(num_assets * [1. / num_assets])

# Optimize for Maximum Sharpe Ratio
result_sharpe = minimize(
    negative_sharpe_ratio,
    initial_weights,
    args=(expected_returns, cov_matrix),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Optimize for Minimum Volatility
result_min_vol = minimize(
    portfolio_volatility,
    initial_weights,
    args=(expected_returns, cov_matrix),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Efficient frontier calculation
target_returns = np.linspace(expected_returns.min(), expected_returns.max(), 50)
efficient_volatilities = []

for target in target_returns:
    constraint = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: portfolio_return(x, expected_returns, cov_matrix) - target}
    ]
    
    result = minimize(
        portfolio_volatility,
        initial_weights,
        args=(expected_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraint
    )
    efficient_volatilities.append(result.fun)

results_df = pd.DataFrame({
    'Asset': tickers,
    'Expected Return': expected_returns,
    'Optimal Weight (Max Sharpe)': result_sharpe.x,
    'Optimal Weight (Min Volatility)': result_min_vol.x
})
print("\nOptimization Results:")
print(results_df)

max_sharpe_return, max_sharpe_vol = portfolio_performance(result_sharpe.x, expected_returns, cov_matrix)
min_vol_return, min_vol_vol = portfolio_performance(result_min_vol.x, expected_returns, cov_matrix)

print(f"\nMaximum Sharpe Ratio Portfolio:")
print(f"Return: {max_sharpe_return:.2%}")
print(f"Volatility: {max_sharpe_vol:.2%}")
print(f"Sharpe Ratio: {(max_sharpe_return - 0.02)/max_sharpe_vol:.2f}")

print(f"\nMinimum Volatility Portfolio:")
print(f"Return: {min_vol_return:.2%}")
print(f"Volatility: {min_vol_vol:.2%}")
print(f"Sharpe Ratio: {(min_vol_return - 0.02)/min_vol_vol:.2f}")

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(np.sqrt(np.diag(cov_matrix)), expected_returns, marker='o', s=100, alpha=0.7, label='Assets')
for i, ticker in enumerate(tickers):
    ax.annotate(ticker, 
                (np.sqrt(np.diag(cov_matrix))[i], expected_returns[i]),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=10)

ax.plot(efficient_volatilities, target_returns, 'b--', linewidth=2, label='Efficient Frontier')
ax.scatter(max_sharpe_vol, max_sharpe_return, marker='*', s=300, c='red', label='Max Sharpe Ratio')
ax.scatter(min_vol_vol, min_vol_return, marker='*', s=300, c='green', label='Min Volatility')

ax.annotate('Max Sharpe', 
            (max_sharpe_vol, max_sharpe_return),
            xytext=(10, 0), 
            textcoords='offset points',
            fontsize=12, 
            weight='bold')

ax.annotate('Min Volatility', 
            (min_vol_vol, min_vol_return),
            xytext=(10, 0), 
            textcoords='offset points',
            fontsize=12, 
            weight='bold')

ax.set_xlabel('Volatility (Standard Deviation)')
ax.set_ylabel('Expected Return')
ax.set_title('Mean-Variance Efficient Frontier')
ax.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.pie(result_sharpe.x, labels=tickers, autopct='%1.1f%%', startangle=90)
ax1.set_title('Optimal Portfolio Weights (Max Sharpe Ratio)')
ax2.pie(result_min_vol.x, labels=tickers, autopct='%1.1f%%', startangle=90)
ax2.set_title('Optimal Portfolio Weights (Min Volatility)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Asset Correlation Matrix')
plt.tight_layout()
plt.show()
