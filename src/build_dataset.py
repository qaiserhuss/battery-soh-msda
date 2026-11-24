"""
Build the model-ready dataset from the NASA discharge CSVs in data/raw (one CSV per battery, one row per measurement:
cycle number, time, voltage, current, temperature, current load, voltage load, capacity).

  - One discharge cycle = one input sequence of 129 points (linear interpolation over time), 4 channels:
    cycle number / 200 (same constant for every battery), voltage, current, temperature.
  - Label: SOH = C / 2.0 Ah (nominal capacity, Eq. 13).
  - Cycles with a recorded capacity <= 0.1 Ah are sensor errors and are removed.
  - Voltage, current and temperature are min-max scaled with statistics from the training data of each split only
    (both labelled sources + the three unlabelled target batteries); the held-out test battery is scaled with the
    same statistics. Labels of the unlabelled target batteries are never saved.

Domains: Condition 2 = #29-#32 (43 C, 4 A; "source1" files), Condition 3 = #45-#48 (4 C, 1 A; "source2" files),
Condition 1 = #05, #06, #07, #18 (24 C, 2 A; target). In each split ("Set"), one Condition 1 battery is the held-out
test battery and the other three are unlabelled target data.

Usage:  python build_dataset.py      (writes data/processed/<Set ...>/*.npy and data/processed/metadata.json)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ_LEN = 129
C_NOMINAL = 2.0
CYCLE_SCALE = 200.0
MIN_CAPACITY = 0.1

SOURCE1 = ["B2/B0029.csv", "B2/B0030.csv", "B2/B0031.csv", "B2/B0032.csv"]
SOURCE2 = ["B3/B0045.csv", "B3/B0046.csv", "B3/B0047.csv", "B3/B0048.csv"]
TARGET = {5: "B1/B0005.csv", 6: "B1/B0006.csv", 7: "B1/B0007.csv", 18: "B1/B0018.csv"}
SETS = {  # test battery -> (folder, unlabelled target file, test data file, test label file)
    5:  ("Set 4 (6,7,18, 5 as test)", "source5to_data.npy",  "b5_data.npy", "B5_test.npy"),
    6:  ("Set 3 (5,7,18, 6 as test)", "source6to_data.npy",  "b6_data.npy", "B6_test.npy"),
    7:  ("Set 2 (5,6,18, 7 as test)", "source7to_data.npy",  "b7_data.npy", "B7_test.npy"),
    18: ("Set 1 (5,6,7, 18 as test)", "source18to_data.npy", "b8_data.npy", "B8_test.npy"),
}


def load_battery(path):
    """Return (X: cycles x SEQ_LEN x 4 [cycle, V, I, T] unscaled, capacity Ah per cycle, info)."""
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = ["cycle", "time", "V", "I", "T", "I_load", "V_load", "capacity"]
    xs, caps, dropped = [], [], []
    for index, (cyc, g) in enumerate(df.groupby("cycle", sort=True), start=1):
        g = g.dropna(subset=["time", "V", "I", "T"]).sort_values("time").drop_duplicates("time")
        cap = float(g.capacity.iloc[0])
        if cap <= MIN_CAPACITY or len(g) < 10:
            dropped.append(index)   # still counted as an experienced cycle
            continue
        t = g.time.values - g.time.values[0]
        grid = np.linspace(0.0, t[-1], SEQ_LEN)
        vit = [np.interp(grid, t, g[c].values) for c in ["V", "I", "T"]]
        xs.append(np.stack([np.full(SEQ_LEN, index, dtype=float)] + vit, axis=1))
        caps.append(cap)
    info = dict(discharge_cycles=index, cycles_kept=len(caps), cycles_dropped=dropped,
                capacity_first=caps[0], capacity_max=max(caps), capacity_min=min(caps))
    return np.asarray(xs, np.float64), np.asarray(caps, np.float32), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "..", "data", "raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "processed"))
    args = ap.parse_args()

    meta = {"seq_len": SEQ_LEN, "channels": ["cycle number / %g" % CYCLE_SCALE, "voltage", "current", "temperature"],
            "label": f"SOH = C / {C_NOMINAL} Ah", "min_capacity_Ah": MIN_CAPACITY, "batteries": {}, "splits": {}}
    cache = {}
    for f in SOURCE1 + SOURCE2 + list(TARGET.values()):
        x, c, info = load_battery(os.path.join(args.raw, f))
        cache[f] = (x, c)
        meta["batteries"][os.path.basename(f)[:-4]] = info
        print(f"{f}: {info['cycles_kept']} cycles kept, {len(info['cycles_dropped'])} dropped, "
              f"capacity {info['capacity_max']:.3f} -> {info['capacity_min']:.3f} Ah")

    cat = lambda files: (np.concatenate([cache[f][0] for f in files]), np.concatenate([cache[f][1] for f in files]))
    s1_x, s1_c = cat(SOURCE1)
    s2_x, s2_c = cat(SOURCE2)

    for b, (folder, pool_name, test_x_name, test_y_name) in SETS.items():
        pool_files = [f for k, f in TARGET.items() if k != b]
        pool_x, _ = cat(pool_files)                      # target labels are never saved
        test_x, test_c = cache[TARGET[b]]
        train = np.concatenate([s1_x, s2_x, pool_x])
        lo, hi = train[:, :, 1:].min(axis=(0, 1)), train[:, :, 1:].max(axis=(0, 1))

        def scale(x):
            out = np.empty_like(x)
            out[:, :, 0] = x[:, :, 0] / CYCLE_SCALE
            out[:, :, 1:] = (x[:, :, 1:] - lo) / (hi - lo)
            return out.astype(np.float32)

        d = os.path.join(args.out, folder)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "source1_data.npy"), scale(s1_x))
        np.save(os.path.join(d, "source1_labels.npy"), (s1_c / C_NOMINAL).astype(np.float32))
        np.save(os.path.join(d, "source2_data.npy"), scale(s2_x))
        np.save(os.path.join(d, "source2_labels.npy"), (s2_c / C_NOMINAL).astype(np.float32))
        np.save(os.path.join(d, pool_name), scale(pool_x))
        np.save(os.path.join(d, test_x_name), scale(test_x))
        np.save(os.path.join(d, test_y_name), (test_c / C_NOMINAL).astype(np.float32))
        meta["splits"][f"#{b:02d}"] = dict(folder=folder, unlabelled_target=[os.path.basename(f)[:-4] for f in pool_files],
                                          n_source1=len(s1_x), n_source2=len(s2_x), n_unlabelled_target=len(pool_x),
                                          n_test=len(test_x), vit_min=lo.tolist(), vit_max=hi.tolist())
        tx = scale(test_x)
        print(f"#{b:02d}: unlabelled target {len(pool_x)}, test {len(test_x)} | test cycle channel "
              f"{tx[:, 0, 0].min():.3f}-{tx[:, 0, 0].max():.3f} | test SOH {test_c.max() / C_NOMINAL:.3f} -> "
              f"{test_c.min() / C_NOMINAL:.3f}")

    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
