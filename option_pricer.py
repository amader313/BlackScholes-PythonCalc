import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt

strike = 105
time_to_expiry = 1
interest_rate = 0.05
volatility = 0.5

def black_scholes_call(S, K, T ,r , sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return(S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))

def black_scholes_put(S, K , T ,r ,sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))

stock_range = np.linspace(80, 120, 100)
calls = [black_scholes_call(s, strike, time_to_expiry, interest_rate, volatility) for s in stock_range]
puts = [black_scholes_put(s, strike, time_to_expiry, interest_rate, volatility) for s in stock_range]

plt.figure(figsize=(10,6))
plt.plot(stock_range, calls, label='Call Price', color='green')
plt.plot(stock_range, puts, label='Put Price', color='red')
plt.axvline(x=strike, color='black', linestyle='--', label=f'Strike Price (${strike})')

plt.xlabel('Stock Price')
plt.ylabel('Option Price')
plt.title(f'Option Prices (Vol: {volatility*100}%, Time : {time_to_expiry}yr)')
plt.legend()
plt.grid(True)
plt.show()
