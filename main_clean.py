import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import datetime
import time

# Function for safer downloads as I was faced a lot of times with rate limits from Yahoo Finance
def safe_download(tickers, start, end, retries=3):
    for attempt in range(retries):
        try:
            data = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False
            )
            return data
        except Exception as e:
            if "Too Many Requests" in str(e):
                print(f"Rate limit hit. Retrying in 10 seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(10)
            else:
                print(f"Download error: {e}")
                break
    raise RuntimeError("Failed to download data after several retries.")

# All of the things have beenwrapped in a function for better organization, this function will get user inputs
def get_user_inputs():
    stocks = []
    for i in range(3):
        ticker = input(f"Enter ticker symbol of stock {i + 1}: ").upper()
        stocks.append(ticker)
    
    try:
        budget = int(input("Enter your investment budget in dollars: "))
    except ValueError:
        raise ValueError("Investment budget must be an integer.")
    
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")
    
    try:
        datetime.datetime.strptime(start_date, "%Y-%m-%d")
        datetime.datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Dates must be in the format YYYY-MM-DD.")
    
    return stocks, budget, start_date, end_date

# Downloads historical data
def fetch_data(stocks, start, end):
    try:
        data = safe_download(stocks, start, end)
        adj_close = data["Adj Close"].dropna(axis=1, how="all")

        sp500 = safe_download("^GSPC", start, end)["Adj Close"].dropna()
        sp500_normalized = (sp500 / sp500.iloc[0])
        return adj_close, sp500_normalized
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        exit()

# Portfolio optimization function
def optimize_portfolio(adj_close):
    returns = adj_close.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(adj_close.columns)
    
    def negative_return(weights):
        return -np.dot(weights, mean_returns)
    
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    initial_guess = np.ones(num_assets) / num_assets
    
    result = minimize(negative_return, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    optimal_weights = result.x
    return optimal_weights, mean_returns, cov_matrix

# Prints allocation summary
def print_allocation(optimal_weights, tickers, mean_returns, cov_matrix, investment_limit, latest_prices):
    port_return = np.dot(optimal_weights, mean_returns)
    port_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    
    print("\nOptimal Portfolio Allocation:")
    for i, ticker in enumerate(tickers):
        print(f"{ticker}: {optimal_weights[i]*100:.2f}%")
    
    print(f"\nExpected Daily Return: {port_return*100:.4f}%")
    print(f"Expected Daily Volatility: {port_volatility*100:.4f}%")
    
    annual_return = port_return * 252
    annual_volatility = port_volatility * np.sqrt(252)
    print(f"Annualized Return: {annual_return*100:.2f}%")
    print(f"Annualized Volatility: {annual_volatility*100:.2f}%")

    allocation_amount = optimal_weights * investment_limit
    num_shares = (allocation_amount / latest_prices).astype(int)
    
    print(f"\nInvestment Plan for ${investment_limit}:")
    for i, ticker in enumerate(tickers):
        print(f"{ticker}: ${allocation_amount[i]:.2f} → {num_shares[i]} shares at ${latest_prices.iloc[i]:.2f}")

# Plots portfolio vs. stocks vs. S&P 500
def plot_performance(adj_close, weights, sp500_norm, investment_limit):
    tickers = adj_close.columns
    normalized_prices = adj_close / adj_close.iloc[0]
    portfolio_value = (normalized_prices * weights).sum(axis=1) * investment_limit
    
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        plt.plot(normalized_prices[ticker] * investment_limit, label=ticker)
    plt.plot(portfolio_value, label="Optimized Portfolio", linestyle="--", linewidth=3, color="black")
    plt.plot(sp500_norm * investment_limit, label="S&P 500", linestyle=":", linewidth=2, color="red")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.title("Portfolio vs Individual Stocks vs S&P 500")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main program entry
def main():
    try:
        stocks, budget, start, end = get_user_inputs()
        adj_close, sp500_norm = fetch_data(stocks, start, end)
        if adj_close.empty:
            print("No valid stock data found.")
            return

        optimal_weights, mean_returns, cov_matrix = optimize_portfolio(adj_close)
        latest_prices = adj_close.iloc[-1]
        print_allocation(optimal_weights, adj_close.columns, mean_returns, cov_matrix, budget, latest_prices)
        plot_performance(adj_close, optimal_weights, sp500_norm, budget)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
