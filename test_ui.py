import yfinance as yf

ticker = yf.Ticker("AXISBANK.NS")
print("--- TEST 1: Income Stmt ---")
print(ticker.income_stmt)

print("\n--- TEST 2: Financials ---")
print(ticker.financials)

print("\n--- TEST 3: Info Dict ---")
try:
    print(ticker.info.get("totalRevenue"))
except Exception as e:
    print("Error:", e)