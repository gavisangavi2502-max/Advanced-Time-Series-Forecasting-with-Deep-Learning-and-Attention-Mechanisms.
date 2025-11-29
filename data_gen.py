
import numpy as np
import pandas as pd

def generate_data(n=1200):
    t = np.arange(n)

    seasonal = 0.5*np.sin(2*np.pi*t/50)
    trend = 0.0008*t
    break_point = int(n*0.6)
    noise = 0.1*np.random.randn(n)

    x1 = seasonal + trend + noise
    x2 = np.cos(0.015*t) + 0.2*np.random.randn(n)

    y = 0.4*x1 + 0.3*x2 + 0.2*np.sin(0.01*t) + 0.2*np.random.randn(n)

    # structural break
    y[break_point:] += 0.8

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data.csv", index=False)
