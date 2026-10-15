import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root, resolved from this file's location

os.chdir(_REPO)


def convergence_stats(val_series, tail_frac=0.15):
    """Check if val loss has plateaued in the last tail_frac of training."""
    n = len(val_series)
    tail = val_series.iloc[int(n * (1 - tail_frac)):]
    if len(tail) < 2 or tail.nunique() == 1:
        return 0.0, 0.0
    slope = np.polyfit(np.arange(len(tail)), tail.values, 1)[0]
    rel_improvement = (tail.iloc[0] - tail.iloc[-1]) / tail.iloc[0]
    return slope, rel_improvement

for pred_ds in ['k562', 'rpe1', 'hepg2', 'jurkat']:
    csv_path = f'STATE/results/{pred_ds}/version_0/metrics.csv'
    if not os.path.exists(csv_path):
        print(f"Skipping {pred_ds}: metrics.csv not found")
        continue

    df = pd.read_csv(csv_path)
    train = df[['step', 'train_loss']].dropna()
    val   = df[['step', 'val_loss']].dropna()

    slope, rel_imp = convergence_stats(val['val_loss'])
    converged = abs(rel_imp) < 0.01  # <1% improvement in last 15% of steps
    status = 'CONVERGED' if converged else 'still declining'

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train['step'], train['train_loss'], lw=0.8, alpha=0.6, color='steelblue', label='train')
    ax.plot(val['step'], val['val_loss'], lw=0.8, alpha=0.8, color='tomato', label='val')
    ax.set_title(
        f'{pred_ds}  |  step {int(train["step"].max())} / 80000  |  '
        f'val={val["val_loss"].iloc[-1]:.4f}  |  [{status}]  |  '
        f'last-15% Δ={rel_imp*100:.1f}%'
    )
    ax.set_xlabel('Step')
    ax.set_ylabel('MMD loss')
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = f'STATE/results/{pred_ds}/loss_curve.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(
        f"{pred_ds}: step {int(train['step'].max())}/80000 | "
        f"train={train['train_loss'].iloc[-1]:.4f}  val={val['val_loss'].iloc[-1]:.4f} | "
        f"last-15% rel_improvement={rel_imp*100:.1f}%  slope={slope:.2e}  → {status}"
    )
