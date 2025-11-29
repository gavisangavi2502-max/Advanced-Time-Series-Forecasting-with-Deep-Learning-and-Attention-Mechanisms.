# Generates synthetic hourly series with trend, seasonality, structural break.
import numpy as np
import pandas as pd, os
def generate_series(n=2500, seed=0):
    np.random.seed(seed)
    t = np.arange(n)
    trend = 0.005 * t
    season = 2.0 * np.sin(2 * np.pi * t / 24)
    noise = 0.6 * np.random.randn(n)
    series = 10 + trend + season + noise
    series[1500:] += 4.0  # structural break
    return pd.DataFrame({'y': series}, index=pd.date_range('2020-01-01', periods=n, freq='H'))
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate_series()
    df.to_csv('data/synthetic_series.csv')
    print('Saved data/synthetic_series.csv', df.shape)
