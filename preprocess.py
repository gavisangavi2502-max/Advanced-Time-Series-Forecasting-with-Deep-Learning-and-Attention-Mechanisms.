
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def add_lags(df, target='y', lags=5):
    for i in range(1, lags+1):
        df[f"{target}_lag{i}"] = df[target].shift(i)
    return df

def load_and_prepare(path="data.csv"):
    df = pd.read_csv(path)
    df = add_lags(df)
    df = df.dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    return df, scaled, scaler
