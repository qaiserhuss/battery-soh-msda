"""
Train and evaluate the multi-source domain adaptation models on the NASA battery data (leave-one-battery-out over
the four Condition 1 batteries #05, #06, #07, #18; five seeds each).

Methods (--method):
  ours_bn        proposed framework as reported in the paper ("Ours")
  ours_bn_nommd  ablation: without the feature-alignment loss (lambda_align = 0)
  ours_bn_noadv  ablation: without the domain-classification loss (lambda_dom = 0)
  ours_bn_noatt  ablation: without attention (fixed weights 0.5 / 0.5 in training and at inference)
  ours_ms        variant: each DSE supervised by its own source labels, losses averaged (Table 7)
  ours           original training setting (DSE batch normalization in training mode during training)
  msdan          MSDAN with its CNN+BiGRU extractor
  msdan_lstm     MSDAN with the LSTM extractor of Ours (backbone-matched comparison)

Each (method, battery, seed) run appends one row to <out>/runs_B<id>.csv and saves the test predictions in
<out>/preds/<method>_B<id>_s<seed>.npy; finished runs are skipped, so the script can be stopped and restarted.

Usage:
    python run_experiment.py --method ours_bn --battery all --seeds 0-4
    python run_experiment.py --method msdan,msdan_lstm --battery 5,7 --seeds 0-4 --threads 3
"""
import argparse
import csv
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "processed")   # written by build_dataset.py
RESULTS = os.path.join(HERE, "..", "results", "main")

# battery id -> (folder, unlabelled target file, test data file, test label file)
SETS = {
    5:  ("Set 4 (6,7,18, 5 as test)", "source5to_data.npy",  "b5_data.npy", "B5_test.npy"),
    6:  ("Set 3 (5,7,18, 6 as test)", "source6to_data.npy",  "b6_data.npy", "B6_test.npy"),
    7:  ("Set 2 (5,6,18, 7 as test)", "source7to_data.npy",  "b7_data.npy", "B7_test.npy"),
    18: ("Set 1 (5,6,7, 18 as test)", "source18to_data.npy", "b8_data.npy", "B8_test.npy"),
}

BATCH_SIZE = 16
LR = 1e-4
MMD_LAMBDA, ADV_LAMBDA = 0.05, 0.05   # lambda_align, lambda_dom in the paper
EPOCHS = {"ours": 500, "ours_bn": 500, "msdan": 100, "msdan_lstm": 100, "ours_ms": 500,
          "ours_bn_nommd": 500, "ours_bn_noadv": 500, "ours_bn_noatt": 500}
# Ablations (Table 5): which term is switched off
ABLATIONS = {"ours_bn_nommd": dict(mmd_lambda=0.0),
             "ours_bn_noadv": dict(adv_lambda=0.0),
             "ours_bn_noatt": dict(fixed_attention=True)}
METHOD_KW = dict(ABLATIONS, ours_ms=dict(own_labels=True))


def load_set(battery):
    folder, tgt, test_x, test_y = SETS[battery]
    d = os.path.join(DATA, folder)
    ld = lambda f: np.load(os.path.join(d, f)).astype(np.float32)
    return dict(
        s1_x=ld("source1_data.npy"), s1_y=ld("source1_labels.npy"),
        s2_x=ld("source2_data.npy"), s2_y=ld("source2_labels.npy"),
        t_x=ld(tgt), test_x=ld(test_x), test_y=ld(test_y),
    )


def make_train_dataset(data):
    s1 = tf.data.Dataset.from_tensor_slices((data["s1_x"], data["s1_y"])) \
        .shuffle(1000).batch(BATCH_SIZE, drop_remainder=True)
    s2 = tf.data.Dataset.from_tensor_slices((data["s2_x"], data["s2_y"])) \
        .shuffle(1000).batch(BATCH_SIZE, drop_remainder=True)
    t = tf.data.Dataset.from_tensor_slices(data["t_x"]) \
        .shuffle(1000).batch(BATCH_SIZE, drop_remainder=True)
    return tf.data.Dataset.zip((s1, s2, t))


def three_class_adv_loss(disc, f1, f2, ft):
    """Cross-entropy of the 3-class domain discriminator (Condition 2 / Condition 3 / target), Eq. (8)."""
    p1, p2, pt = disc(f1), disc(f2), disc(ft)
    labels = tf.concat([
        tf.zeros((tf.shape(p1)[0],), dtype=tf.int32),
        tf.ones((tf.shape(p2)[0],), dtype=tf.int32),
        2 * tf.ones((tf.shape(pt)[0],), dtype=tf.int32),
    ], axis=0)
    return tf.keras.losses.SparseCategoricalCrossentropy()(labels, tf.concat([p1, p2, pt], axis=0))


# ---------------------------------------------------------------- Ours
def wasserstein_distance(source, target):
    """Feature-alignment loss, Eq. (7): mean distance of each source feature to its nearest target feature."""
    distance_matrix = tf.norm(source[:, None] - target[None, :], axis=-1)
    return tf.reduce_mean(tf.reduce_min(distance_matrix, axis=1))


class AttentionWeighting(tf.keras.layers.Layer):
    """Relevance score of one source from [mean source feature ; mean target feature], Eq. (10)."""
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(1, activation='sigmoid')

    def call(self, source_features, target_features):
        source_mean = tf.reduce_mean(source_features, axis=0, keepdims=True)
        target_mean = tf.reduce_mean(target_features, axis=0, keepdims=True)
        combined = tf.concat([source_mean, target_mean], axis=-1)
        return self.dense2(self.dense1(combined))


def make_lstm_extractor(timesteps=129, features=4, lstm_units=128, num_lstm_layers=2, dense_units=256):
    """Common feature extractor (CFE): B x 129 x 4 -> B x 256."""
    inp = layers.Input(shape=(timesteps, features))
    x = inp
    for _ in range(num_lstm_layers):
        x = layers.LSTM(lstm_units, return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    return models.Model(inputs=inp, outputs=x)


def make_ours_head():
    """Domain-specific extractor / regressor (DSE): B x 256 -> B x 1."""
    return models.Sequential([
        layers.Input(shape=(256,)),
        layers.Dense(640, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(320, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(1),
    ])


def make_ours_discriminator():
    return models.Sequential([
        layers.Input(shape=(256,)),
        layers.Dense(640, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(320, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(3, activation='softmax'),
    ])


def call_head_bn_inference(head, f):
    """Head forward pass with BatchNorm in inference mode and Dropout active (training)."""
    x = f
    for layer in head.layers:
        if isinstance(layer, layers.BatchNormalization):
            x = layer(x, training=False)
        elif isinstance(layer, layers.Dropout):
            x = layer(x, training=True)
        else:
            x = layer(x)
    return x


def run_ours(data, epochs, head_bn_training=True, mmd_lambda=MMD_LAMBDA, adv_lambda=ADV_LAMBDA,
             fixed_attention=False, own_labels=False):
    # head_bn_training=False is the setting reported in the paper (ours_bn): the DSEs' BatchNorm uses its stored
    # statistics during training as well (as the extractor does), while Dropout stays active.
    extractor = make_lstm_extractor()
    heads = [make_ours_head() for _ in range(2)]
    attention = AttentionWeighting()
    disc = make_ours_discriminator()
    opt, disc_opt = tf.keras.optimizers.Adam(LR), tf.keras.optimizers.Adam(LR)

    def calculate_weights(f1, f2, ft):
        if fixed_attention:
            return tf.constant(0.5), tf.constant(0.5)
        w1, w2 = attention(f1, ft), attention(f2, ft)
        total = w1 + w2 + 1e-8
        return w1 / total, w2 / total

    # The extractor is always called with training=False: its BatchNorm uses stored statistics and its Dropout is
    # inactive; its weights are still trained.
    @tf.function
    def train_step(s1_x, s1_y, s2_x, s2_y, t_x):
        with tf.GradientTape(persistent=True) as tape:
            f1 = extractor(s1_x, training=False)
            f2 = extractor(s2_x, training=False)
            ft = extractor(t_x, training=False)
            if head_bn_training:
                p1 = heads[0](f1, training=True)
                p2 = heads[1](f2, training=True)
            else:
                p1 = call_head_bn_inference(heads[0], f1)
                p2 = call_head_bn_inference(heads[1], f2)
            w1, w2 = calculate_weights(f1, f2, ft)
            mse = tf.keras.losses.MeanSquaredError()
            if own_labels:
                reg_loss = 0.5 * (mse(s1_y, p1) + mse(s2_y, p2))
            else:
                reg_loss = mse(s1_y, w1 * p1 + w2 * p2)   # Eq. (9)
            mmd_val = wasserstein_distance(f1, ft)
            disc_loss = three_class_adv_loss(disc, f1, f2, ft)
            total_loss = reg_loss + mmd_lambda * mmd_val + adv_lambda * disc_loss   # Eq. (12)
        all_vars = (extractor.trainable_variables
                    + [v for h in heads for v in h.trainable_variables]
                    + attention.trainable_variables)
        grads = tape.gradient(total_loss, all_vars)
        opt.apply_gradients([(g, v) for g, v in zip(grads, all_vars) if g is not None])
        disc_opt.apply_gradients(zip(tape.gradient(disc_loss, disc.trainable_variables),
                                     disc.trainable_variables))
        return reg_loss, mmd_val, disc_loss, total_loss

    history = train_loop(train_step, make_train_dataset(data), epochs)

    # Reported prediction: equal 0.5/0.5 weights over the two DSEs; the attention-weighted one is saved as well.
    ft = extractor(data["test_x"], training=False)
    p1 = heads[0](ft, training=False).numpy().ravel()
    p2 = heads[1](ft, training=False).numpy().ravel()
    # Attention weights at test time: all source samples vs. the unlabelled cycles of the test battery.
    f1 = extractor(data["s1_x"], training=False)
    f2 = extractor(data["s2_x"], training=False)
    a1, a2 = (float(w.numpy().squeeze()) for w in calculate_weights(f1, f2, ft))
    return {"pred": 0.5 * p1 + 0.5 * p2, "pred_attention": a1 * p1 + a2 * p2,
            "pred_head1": p1, "pred_head2": p2, "alpha1": a1, "alpha2": a2, "history": history}


# ---------------------------------------------------------------- MSDAN
class CBiGRUFeatureExtractor(layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv1D(64, 5, strides=2, activation='relu')
        self.pool1 = layers.MaxPool1D(2)
        self.conv2 = layers.Conv1D(64, 5, strides=2, activation='relu')
        self.pool2 = layers.MaxPool1D(2)
        self.flatten = layers.Flatten()
        self.bigru = layers.Bidirectional(layers.GRU(128, return_sequences=False))

    def call(self, x):
        c = self.flatten(self.pool2(self.conv2(self.pool1(self.conv1(x)))))
        g = self.bigru(x)
        return tf.concat([c, g], axis=-1)


def linear_mmd(x, y):
    xx = tf.matmul(x, x, transpose_b=True)
    yy = tf.matmul(y, y, transpose_b=True)
    xy = tf.matmul(x, y, transpose_b=True)
    return tf.reduce_mean(xx) + tf.reduce_mean(yy) - 2.0 * tf.reduce_mean(xy)


def run_msdan(data, epochs, backbone="cnn_bigru"):
    # backbone="lstm": MSDAN with the LSTM extractor of Ours, called with training=False exactly as in Ours;
    # generators, estimators, MMD, discriminator and averaging unchanged.
    if backbone == "lstm":
        lstm = make_lstm_extractor()
        extractor = lambda x, training=False: lstm(x, training=False)
        extractor.trainable_variables = lstm.trainable_variables
    else:
        extractor = CBiGRUFeatureExtractor()
    generators = [models.Sequential([layers.Dense(128, activation='relu'),
                                     layers.Dense(128, activation='relu')]) for _ in range(2)]
    estimators = [models.Sequential([layers.Dense(64, activation='relu'),
                                     layers.Dense(64, activation='relu'),
                                     layers.Dense(1)]) for _ in range(2)]
    disc = models.Sequential([layers.Dense(64, activation='relu'),
                              layers.Dense(64, activation='relu'),
                              layers.Dense(3, activation='softmax')])
    opt, disc_opt = tf.keras.optimizers.Adam(LR), tf.keras.optimizers.Adam(LR)

    @tf.function
    def train_step(s1_x, s1_y, s2_x, s2_y, t_x):
        with tf.GradientTape(persistent=True) as tape:
            cf = [extractor(s1_x, training=True), extractor(s2_x, training=True)]
            cf_t = extractor(t_x, training=True)
            preds = [e(g(c, training=True), training=True)
                     for c, g, e in zip(cf, generators, estimators)]
            reg_loss = tf.keras.losses.MeanSquaredError()(s1_y, tf.reduce_mean(tf.stack(preds), axis=0))
            mmd_val = tf.add_n([linear_mmd(c, cf_t) for c in cf])
            adv_loss = three_class_adv_loss(lambda f: disc(f, training=True), cf[0], cf[1], cf_t)
            total_loss = reg_loss + MMD_LAMBDA * mmd_val + ADV_LAMBDA * adv_loss
        train_vars = (extractor.trainable_variables
                      + [v for g in generators for v in g.trainable_variables]
                      + [v for e in estimators for v in e.trainable_variables])
        opt.apply_gradients(zip(tape.gradient(total_loss, train_vars), train_vars))
        disc_opt.apply_gradients(zip(tape.gradient(adv_loss, disc.trainable_variables),
                                     disc.trainable_variables))
        return reg_loss, mmd_val, adv_loss, total_loss

    history = train_loop(train_step, make_train_dataset(data), epochs)

    cf_test = extractor(data["test_x"], training=False)
    p1, p2 = (e(g(cf_test, training=False), training=False).numpy().ravel()
              for g, e in zip(generators, estimators))
    return {"pred": 0.5 * p1 + 0.5 * p2, "pred_head1": p1, "pred_head2": p2, "history": history}


# ---------------------------------------------------------------- driver
def train_loop(train_step, dataset, epochs):
    history = np.zeros((epochs, 4), dtype=np.float32)
    for epoch in range(epochs):
        sums, n = np.zeros(4), 0
        for (s1_x, s1_y), (s2_x, s2_y), t_x in dataset:
            sums += [v.numpy() for v in train_step(s1_x, s1_y, s2_x, s2_y, t_x)]
            n += 1
        history[epoch] = sums / n
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    epoch {epoch + 1}/{epochs}  reg={history[epoch, 0]:.4f} "
                  f"align={history[epoch, 1]:.4f} dom={history[epoch, 2]:.4f}", flush=True)
    return history


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


FIELDS = ["method", "battery", "seed", "epochs", "RMSE", "MAE",
          "RMSE_attention", "MAE_attention", "RMSE_head1", "RMSE_head2",
          "alpha1", "alpha2", "final_reg_loss", "seconds"]


def done_runs(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(r["method"], int(r["battery"]), int(r["seed"]), int(r["epochs"]))
                for r in csv.DictReader(f)}


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="ours_bn", help="one of " + ", ".join(EPOCHS) + ", or a comma list")
    ap.add_argument("--battery", default="all", help="5, 6, 7, 18, a comma list, or all")
    ap.add_argument("--seeds", default="0-4", help="e.g. 0-4 or 0,1,2")
    ap.add_argument("--epochs", type=int, default=None, help="override (for smoke tests only)")
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--threads", type=int, default=0, help="TF intra-op threads (0 = TF default)")
    args = ap.parse_args()
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    methods = args.method.split(",")
    assert all(m in EPOCHS for m in methods), f"unknown method in {methods}"
    batteries = list(SETS) if args.battery == "all" else [int(b) for b in args.battery.split(",")]
    seeds = parse_seeds(args.seeds)

    os.makedirs(os.path.join(args.out, "preds"), exist_ok=True)

    for battery in batteries:
        # one file per battery so batteries can run as parallel processes
        runs_csv = os.path.join(args.out, f"runs_B{battery}.csv")
        finished = done_runs(runs_csv)
        data = load_set(battery)
        np.save(os.path.join(args.out, "preds", f"truth_B{battery}.npy"), data["test_y"])
        for method in methods:
            epochs = args.epochs or EPOCHS[method]
            for seed in seeds:
                if (method, battery, seed, epochs) in finished:
                    print(f"skip {method} B{battery} seed {seed} (done)")
                    continue
                print(f"== {method} B{battery} seed {seed} ({epochs} epochs)", flush=True)
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)  # python, numpy and tf seeds
                t0 = time.time()
                if method.startswith("msdan"):
                    out = run_msdan(data, epochs, backbone="lstm" if method == "msdan_lstm" else "cnn_bigru")
                else:
                    out = run_ours(data, epochs, head_bn_training=(method == "ours"),
                                   **METHOD_KW.get(method, {}))
                y = data["test_y"]
                row = dict(method=method, battery=battery, seed=seed, epochs=epochs,
                           RMSE=rmse(y, out["pred"]), MAE=mae(y, out["pred"]),
                           RMSE_attention=rmse(y, out["pred_attention"]) if "pred_attention" in out else "",
                           MAE_attention=mae(y, out["pred_attention"]) if "pred_attention" in out else "",
                           RMSE_head1=rmse(y, out["pred_head1"]), RMSE_head2=rmse(y, out["pred_head2"]),
                           alpha1=out.get("alpha1", ""), alpha2=out.get("alpha2", ""),
                           final_reg_loss=float(out["history"][-1, 0]),
                           seconds=round(time.time() - t0, 1))
                tag = f"{method}_B{battery}_s{seed}" + ("" if epochs == EPOCHS[method] else f"_e{epochs}")
                np.save(os.path.join(args.out, "preds", tag + ".npy"), out["pred"])
                np.save(os.path.join(args.out, "preds", tag + "_history.npy"), out["history"])
                if "pred_attention" in out:
                    np.save(os.path.join(args.out, "preds", tag + "_attention.npy"), out["pred_attention"])
                new = not os.path.exists(runs_csv)
                with open(runs_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                print(f"   RMSE={row['RMSE']:.4f} MAE={row['MAE']:.4f} ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
