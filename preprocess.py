import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib, os
def fit_scaler(series, train_end_idx, save_path='data/scaler.save'):
    scaler = StandardScaler()
    train_vals = series.iloc[:train_end_idx].values.reshape(-1,1)
    scaler.fit(train_vals)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(scaler, save_path)
    return scaler
def transform_with_scaler(series, scaler):
    arr = scaler.transform(series.values.reshape(-1,1)).reshape(-1)
    return pd.Series(arr, index=series.index, name=series.name)
if __name__ == '__main__':
    df = pd.read_csv('data/synthetic_series.csv', index_col=0, parse_dates=True)
    scaler = fit_scaler(df['y'], train_end_idx=1200)
    s = transform_with_scaler(df['y'], scaler)
    s.to_csv('data/synthetic_series_scaled_sample.csv', header=True)
    print('Wrote sample scaled file.')
