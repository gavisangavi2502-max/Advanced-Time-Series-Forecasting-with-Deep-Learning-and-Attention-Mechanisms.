
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_sarimax(path="data.csv"):
    df = pd.read_csv(path)
    y = df['y']
    model = SARIMAX(y, order=(2,0,2), seasonal_order=(1,0,1,50))
    res = model.fit(disp=False)
    preds = res.forecast(50)
    preds.to_csv("sarimax_preds.csv")
