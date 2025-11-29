# Time Series Forecasting Project (v2)
This improved project includes:
- Data generation (`data_gen.py`)
- Preprocessing with proper train-only scaling (`preprocess.py`)
- LSTM+Attention model (`model_lstm_attention.py`)
- Transformer model (`model_transformer.py`)
- SARIMAX baseline integrated in CV (`baseline_sarimax.py`)
- Training & evaluation with rolling-window CV and backtesting (`train.py`)
- Optuna hyperparameter search (`optuna_search.py`)
- Evaluation script (`evaluate.py`) producing metrics and plots
- Attention interpretation notes (`analysis/attention_interpretation.md`)
- Utilities (`utils.py`)
- `requirements.txt` and `run_all.sh`

Quick run:
1. pip install -r requirements.txt
2. python data_gen.py
3. python train.py --model lstm
4. python evaluate.py
