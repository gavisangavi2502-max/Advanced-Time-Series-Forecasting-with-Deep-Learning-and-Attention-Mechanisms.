import statsmodels.api as sm
def sarimax_forecast(train, steps=24, order=(1,0,1), seasonal_order=(1,1,1,24)):
    model = sm.tsa.SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps)
    return pred.predicted_mean
