import numpy as np
def calculate_sma(prices, n=10):
    sma = np.zeros_like(prices)
    for i in range(n, len(prices)):
        sma[i] = prices[i-n:i].mean()
    return sma
