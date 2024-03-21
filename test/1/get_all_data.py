import yfinance as yf

# Get all available tickers
all_tickers = yf.Tickers()

# Get all available tickers symbols
ticker_symbols = all_tickers.Tickers

# Extract ticker symbols from the list
ticker_symbols_list = ticker_symbols.keys()

# Print the list of ticker symbols
for ticker_symbol in ticker_symbols_list:
    print(ticker_symbol)


# sp500_tickers = yf.download("^GSPC", period="1d")["symbols"]

# print(f"Number of S&P 500 constituents: {len(sp500_tickers)}")
# print("Sample:", sp500_tickers[:5])  # Print first 5 symbols
