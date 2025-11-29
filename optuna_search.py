import optuna, numpy as np
from train import train_lstm_single
from sklearn.metrics import mean_absolute_error
def run_search(Xtr, ytr, Xv, yv, seq_len, horizon, device, n_trials=12):
    def objective(trial):
        params = {
            'hidden_dim': trial.suggest_int('hidden_dim', 16, 128),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [32,64]),
            'epochs': 5
        }
        preds,_,_ = train_lstm_single(Xtr,ytr,Xv,yv,seq_len,horizon,device,params, do_train=True)
        return mean_absolute_error(yv[:,0], preds[:,0])
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
