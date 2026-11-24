"""
No-adaptation baselines: one LSTM extractor + one regression head (same layers and training setting as Ours:
extractor called with training=False, head BatchNorm in inference mode, Dropout active), Adam 1e-4, batch 16,
MSE, 100 epochs. No target data is used.

  single_s1   train on Condition 2 (#29-#32) only   ("Single Source" in the paper)
  single_s2   train on Condition 3 (#45-#48) only   ("Single Best": the better single source on every test battery)
  combined    train on Condition 2 + Condition 3 merged into one labelled set ("Source Combined")

Note: "Single Best" is chosen by looking at the test errors, so it is an optimistic baseline.

Usage:
    python run_baselines.py --battery all --seeds 0-4
"""
import argparse
import csv
import os
import time

import numpy as np
import tensorflow as tf

import run_experiment as rx

OUT = os.path.join(rx.HERE, "..", "results", "baselines")
METHODS = ["single_s1", "single_s2", "combined"]
EPOCHS = 100
FIELDS = ["method", "battery", "seed", "epochs", "RMSE", "MAE", "pred_min", "pred_max", "corr",
          "final_loss", "seconds"]


def training_set(data, method):
    if method == "single_s1":
        return data["s1_x"], data["s1_y"]
    if method == "single_s2":
        return data["s2_x"], data["s2_y"]
    return np.concatenate([data["s1_x"], data["s2_x"]]), np.concatenate([data["s1_y"], data["s2_y"]])


def run_baseline(data, method, epochs):
    x, y = training_set(data, method)
    extractor, head = rx.make_lstm_extractor(), rx.make_ours_head()
    opt = tf.keras.optimizers.Adam(rx.LR)
    mse = tf.keras.losses.MeanSquaredError()
    variables = extractor.trainable_variables + head.trainable_variables

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            loss = mse(yb, rx.call_head_bn_inference(head, extractor(xb, training=False)))
        opt.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss

    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1000).batch(rx.BATCH_SIZE, drop_remainder=True)
    history = []
    for epoch in range(epochs):
        history.append(float(np.mean([float(train_step(xb, yb)) for xb, yb in ds])))
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    epoch {epoch + 1}/{epochs}  loss={history[-1]:.4f}", flush=True)
    pred = head(extractor(data["test_x"], training=False), training=False).numpy().ravel()
    return pred, np.array(history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="all")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    os.makedirs(os.path.join(args.out, "preds"), exist_ok=True)

    batteries = list(rx.SETS) if args.battery == "all" else [int(b) for b in args.battery.split(",")]
    for battery in batteries:
        runs_csv = os.path.join(args.out, f"baselines_B{battery}.csv")
        finished = set()
        if os.path.exists(runs_csv):
            with open(runs_csv, newline="") as f:
                finished = {(r["method"], int(r["seed"]), int(r["epochs"])) for r in csv.DictReader(f)}
        data = rx.load_set(battery)
        np.save(os.path.join(args.out, "preds", f"truth_B{battery}.npy"), data["test_y"])
        for method in args.methods.split(","):
            for seed in rx.parse_seeds(args.seeds):
                if (method, seed, args.epochs) in finished:
                    print(f"skip {method} B{battery} seed {seed} (done)")
                    continue
                print(f"== {method} B{battery} seed {seed} ({args.epochs} epochs)", flush=True)
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                t0 = time.time()
                pred, history = run_baseline(data, method, args.epochs)
                y = data["test_y"].ravel()
                row = dict(method=method, battery=battery, seed=seed, epochs=args.epochs,
                           RMSE=rx.rmse(y, pred), MAE=rx.mae(y, pred), pred_min=float(pred.min()),
                           pred_max=float(pred.max()), corr=float(np.corrcoef(pred, y)[0, 1]),
                           final_loss=float(history[-1]), seconds=round(time.time() - t0, 1))
                tag = f"{method}_B{battery}_s{seed}" + ("" if args.epochs == EPOCHS else f"_e{args.epochs}")
                np.save(os.path.join(args.out, "preds", tag + ".npy"), pred)
                np.save(os.path.join(args.out, "preds", tag + "_history.npy"), history)
                new = not os.path.exists(runs_csv)
                with open(runs_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                print(f"   RMSE={row['RMSE']:.4f} MAE={row['MAE']:.4f} corr={row['corr']:.2f} "
                      f"pred {row['pred_min']:.2f}-{row['pred_max']:.2f} ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
