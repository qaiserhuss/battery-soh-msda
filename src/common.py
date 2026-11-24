"""Shared helpers for tables.py and figures.py: load the per-run results and the permutation test."""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
BATTERIES = (5, 6, 7, 18)
EPOCHS = {"single_s1": 100, "single_s2": 100, "combined": 100, "msdan": 100, "msdan_lstm": 100,
          "ours": 500, "ours_bn": 500, "ours_ms": 500, "ours_bn_nommd": 500, "ours_bn_noadv": 500, "ours_bn_noatt": 500}
NAMES = {"single_s1": "Single Source (Condition 2)", "single_s2": "Single Best (Condition 3)",
         "combined": "Source Combined", "msdan": "MSDAN (CNN+BiGRU)", "msdan_lstm": "MSDAN (LSTM backbone)",
         "ours_bn": "Ours", "ours_bn_nommd": "w/o alignment loss", "ours_bn_noadv": "w/o domain-classification loss",
         "ours_bn_noatt": "w/o attention", "ours_ms": "Own labels per DSE", "ours": "Ours, original training setting"}
FOLDER = {m: ("baselines" if m in ("single_s1", "single_s2", "combined") else "main") for m in EPOCHS}


def load():
    """One row per (method, battery, seed) with the full number of epochs."""
    main = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "main", "runs_B*.csv"))])
    base = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "baselines", "baselines_B*.csv"))])
    df = pd.concat([main, base], ignore_index=True)
    df = df[df.epochs == df.method.map(EPOCHS)]
    return df.drop_duplicates(["method", "battery", "seed"], keep="last")


def per_battery(df, method, metric):
    """{battery: array over seeds}"""
    d = df[df.method == method].sort_values("seed")
    return {b: d[d.battery == b][metric].values.astype(float) for b in BATTERIES}


def truth(b):
    return np.load(os.path.join(RES, "main", "preds", f"truth_B{b}.npy")).ravel()


def predictions(method, b, seeds=range(5)):
    """Saved test predictions of one method and test battery, one array per seed."""
    return [np.load(os.path.join(RES, FOLDER[method], "preds", f"{method}_B{b}_s{s}.npy")).ravel() for s in seeds]


def perm_p(groups, n=20000, seed=0):
    """Two-sided permutation test on per-run errors, stratified by test battery.
    groups = [(errors of method A, errors of method B) for each battery]; statistic = mean difference over batteries."""
    rng = np.random.default_rng(seed)
    stat = lambda g: np.mean([a.mean() - r.mean() for a, r in g])
    obs, count = stat(groups), 0
    for _ in range(n):
        perm = []
        for a, r in groups:
            z = rng.permutation(np.concatenate([a, r]))
            perm.append((z[:len(a)], z[len(a):]))
        count += abs(stat(perm)) >= abs(obs)
    return (count + 1) / (n + 1)
