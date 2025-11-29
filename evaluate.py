import numpy as np, pandas as pd, os, matplotlib.pyplot as plt
def main():
    out = 'outputs/metrics.csv'
    if not os.path.exists(out):
        print('Run train.py first to produce outputs/metrics.csv')
        return
    df = pd.read_csv(out)
    print(df)
    # simple plot of metrics
    df.plot(x='split', y=['mae','rmse'], kind='bar', figsize=(8,4))
    plt.tight_layout(); plt.savefig('outputs/metrics_plot.png')
    print('Saved outputs/metrics_plot.png')
if __name__ == '__main__': main()
