import os, argparse, numpy as np, pandas as pd, torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from utils import RollingWindowSplitter, create_sequences
from preprocess import fit_scaler, transform_with_scaler
from model_lstm_attention import LSTMWithAttention
from model_transformer import TimeSeriesTransformer
from baseline_sarimax import sarimax_forecast
import optuna
def train_lstm_single(Xtr,ytr,Xv,yv,seq_len,horizon,device,params, do_train=True):
    model = LSTMWithAttention(input_dim=1, hidden_dim=params['hidden_dim'], num_layers=params['num_layers'], horizon=horizon).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'])
    loss_fn = torch.nn.MSELoss()
    if do_train:
        tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        loader = DataLoader(tr_ds, batch_size=params['batch_size'], shuffle=True)
        for epoch in range(params['epochs']):
            model.train()
            for xb,yb in loader:
                xb,yb = xb.to(device), yb.to(device)
                pred,_ = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds, att = model(torch.from_numpy(Xv).to(device))
    return preds.cpu().numpy(), att.cpu().numpy(), model
def main(args):
    os.makedirs('outputs', exist_ok=True)
    df = pd.read_csv('data/synthetic_series.csv', index_col=0, parse_dates=True)
    splitter = RollingWindowSplitter(n_splits=3)
    splits = splitter.split(df)
    seq_len, horizon = args.seq_len, args.horizon
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    for i,(train, val, test) in enumerate(splits):
        print('Running split', i)
        # Fit scaler only on train portion to avoid leakage
        scaler_path = f'data/scaler_split_{i}.save'
        scaler = fit_scaler(train['y'], train_end_idx=len(train), save_path=scaler_path)
        train_s = transform_with_scaler(train['y'], scaler)
        val_s = transform_with_scaler(val['y'], scaler)
        test_s = transform_with_scaler(test['y'], scaler)
        Xtr,ytr = create_sequences(train_s, seq_len=seq_len, horizon=horizon)
        Xv,yv = create_sequences(val_s, seq_len=seq_len, horizon=horizon)
        Xt,yt = create_sequences(test_s, seq_len=seq_len, horizon=horizon)
        if args.model == 'lstm':
            # quick default params; user can run optuna_search to get better ones
            params = {'hidden_dim':64,'num_layers':2,'lr':1e-3,'batch_size':64,'epochs':8}
            preds, att, model = train_lstm_single(Xtr,ytr,Xt,yt,seq_len,horizon,device,params, do_train=True)
            np.save(f'outputs/att_split_{i}.npy', att)
            np.save(f'outputs/preds_split_{i}.npy', preds)
        elif args.model == 'transformer':
            model = TimeSeriesTransformer(input_dim=1, d_model=64, nhead=4, num_layers=2, seq_len=seq_len, horizon=horizon).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = torch.nn.MSELoss()
            tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
            loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
            for epoch in range(6):
                model.train()
                for xb,yb in loader:
                    xb,yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = loss_fn(out, yb)
                    opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                preds = model(torch.from_numpy(Xt).to(device)).cpu().numpy()
            np.save(f'outputs/preds_split_{i}.npy', preds)
        else:
            # SARIMAX baseline - use combined train+val history to forecast test horizon step-by-step
            history = pd.concat([train, val])
            # forecast step-by-step in original scale (so use raw y)
            mean = sarimax_forecast(history, steps=len(test))
            np.save(f'outputs/preds_split_{i}.npy', mean.values[:len(test)])
        # Evaluate first horizon step consistently
        preds = np.load(f'outputs/preds_split_{i}.npy')
        if preds.ndim == 2:
            pred0 = preds[:,0]
            true0 = yt[:,0]  # scaled test first horizon
        else:
            pred0 = preds[:len(test)]
            true0 = test['y'].values[:len(pred0)]
        # if model produced scaled predictions, invert not implemented here; compute metrics on available scale
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(true0, pred0)
        rmse = mean_squared_error(true0, pred0, squared=False)
        results.append({'split':i,'mae':float(mae),'rmse':float(rmse)})
    pd.DataFrame(results).to_csv('outputs/metrics.csv', index=False)
    print('Saved outputs/metrics.csv')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['lstm','transformer','sarimax'], default='lstm')
    parser.add_argument('--seq_len', type=int, default=48)
    parser.add_argument('--horizon', type=int, default=24)
    args = parser.parse_args()
    main(args)
