"""
Print all result tables of the paper (Tables 3-7) and the numbers quoted in the text, from the saved per-run
results in results/. Also written to results/tables.md. No training.

Usage:  python tables.py
"""
import os

import numpy as np

from common import BATTERIES, NAMES, RES, load, per_battery, perm_p, predictions, truth

LABEL_SPLIT = 0.78   # lower end of the SOH range of the Condition 2 labels
CALIB_K = 5          # one-point calibration: first K cycles of the test battery


def ms(x):
    return f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}"


def avg(d):
    return np.mean([d[b].mean() for b in BATTERIES])


def p_str(p):
    return "< 0.01" if p < 0.01 else f"{p:.2f}"


def main():
    df = load()
    out = []
    say = lambda s="": (print(s), out.append(s))
    rmse = {m: per_battery(df, m, "RMSE") for m in NAMES}
    mae = {m: per_battery(df, m, "MAE") for m in NAMES}
    for m in NAMES:
        assert all(len(rmse[m][b]) == 5 for b in BATTERIES), f"{m}: expected 5 seeds per battery"
    p = {m: perm_p([(rmse[m][b], rmse["ours_bn"][b]) for b in BATTERIES]) for m in NAMES if m != "ours_bn"}

    def table(title, methods):
        say(f"\n## {title}\n")
        say("| Method | " + " | ".join(f"#{b:02d}" for b in BATTERIES) + " | Avg. RMSE | Avg. MAE | p vs Ours |")
        say("|---|" + "---|" * (len(BATTERIES) + 3))
        for m in methods:
            say(f"| {NAMES[m]} | " + " | ".join(ms(rmse[m][b]) for b in BATTERIES)
                + f" | {avg(rmse[m]):.3f} | {avg(mae[m]):.3f} | {p_str(p[m]) if m in p else '–'} |")

    say("# Results (RMSE / MAE in SOH = C / 2.0 Ah; mean ± std over 5 seeds; p: stratified permutation test, RMSE)")
    table("Table 3: comparison with the baselines and MSDAN",
          ["single_s1", "single_s2", "combined", "msdan", "msdan_lstm", "ours_bn"])
    say("\n## Table R3: MAE per test battery\n")
    say("| Method | " + " | ".join(f"#{b:02d}" for b in BATTERIES) + " | Avg. MAE |")
    say("|---|" + "---|" * (len(BATTERIES) + 1))
    for m in ["single_s1", "single_s2", "combined", "msdan", "msdan_lstm", "ours_bn"]:
        say(f"| {NAMES[m]} | " + " | ".join(ms(mae[m][b]) for b in BATTERIES) + f" | {avg(mae[m]):.3f} |")
    table("Table 4: backbone-matched comparison", ["ours_bn", "msdan_lstm", "msdan"])
    table("Table 5: ablation", ["ours_bn", "ours_bn_nommd", "ours_bn_noadv", "ours_bn_noatt"])

    say("\n## Table 6: attention weights at test time (Ours)\n")
    say("| Test battery | w1 (Condition 2) [min-max] | w2 | RMSE DSE 1 | RMSE DSE 2 | RMSE equal weights | RMSE attention |")
    say("|---|---|---|---|---|---|---|")
    diffs = []
    for b in BATTERIES:
        d = df[(df.method == "ours_bn") & (df.battery == b)]
        a = d.alpha1.astype(float).values
        diffs += list(np.abs(d.RMSE.values - d.RMSE_attention.astype(float).values))
        say(f"| #{b:02d} | {ms(a)} [{a.min():.3f}-{a.max():.3f}] | {1 - a.mean():.3f} | "
            f"{d.RMSE_head1.mean():.3f} | {d.RMSE_head2.mean():.3f} | {d.RMSE.mean():.3f} | "
            f"{d.RMSE_attention.astype(float).mean():.3f} |")
    say(f"\nLargest difference between equal-weight and attention-weighted RMSE in a single run: {max(diffs):.4f}")

    say("\n## Table 7: source labels used in the regression loss (RMSE split at SOH 0.78, pooled over all runs)\n")
    say("| Regression loss | " + " | ".join(f"#{b:02d}" for b in BATTERIES)
        + " | Avg. RMSE | Avg. MAE | SOH >= 0.78 | SOH < 0.78 | p vs Ours |")
    say("|---|" + "---|" * (len(BATTERIES) + 5))
    for m in ("ours_bn", "ours_ms", "msdan"):
        hi, lo = [], []
        for b in BATTERIES:
            t = truth(b)
            for pr in predictions(m, b):
                hi.append((pr - t)[t >= LABEL_SPLIT])
                lo.append((pr - t)[t < LABEL_SPLIT])
        r = lambda e: np.sqrt(np.mean(np.concatenate(e) ** 2))
        say(f"| {NAMES[m]} | " + " | ".join(ms(rmse[m][b]) for b in BATTERIES)
            + f" | {avg(rmse[m]):.3f} | {avg(mae[m]):.3f} | {r(hi):.3f} | {r(lo):.3f} | "
            + (p_str(p[m]) if m in p else "–") + " |")

    say("\n## Numbers quoted in the text\n")
    o, c, s1, s2 = avg(rmse["ours_bn"]), avg(rmse["combined"]), avg(rmse["single_s1"]), avg(rmse["single_s2"])
    so = [rmse["ours_bn"][b].std(ddof=1) for b in BATTERIES]
    ss = [rmse["single_s2"][b].std(ddof=1) for b in BATTERIES]
    say(f"- error reduction vs Source Combined: {(c - o) / c:.0%}; vs Single Source (Condition 2): {(s1 - o) / s1:.0%}")
    say(f"- expected RMSE when choosing one of the two sources without target labels: {(s1 + s2) / 2:.3f}")
    say(f"- seed std of RMSE: Ours {min(so):.3f}-{max(so):.3f} (mean {np.mean(so):.3f}); Single Best "
        f"{min(ss):.3f}-{max(ss):.3f} (mean {np.mean(ss):.3f}); ratio of means {np.mean(ss) / np.mean(so):.1f}")
    say("- test batteries where Ours has a lower RMSE than MSDAN (CNN+BiGRU): "
        + ", ".join(f"#{b:02d}" for b in BATTERIES if rmse["ours_bn"][b].mean() < rmse["msdan"][b].mean()))
    say(f"- Ours in Ah: RMSE {2 * o:.3f} Ah, MAE {2 * avg(mae['ours_bn']):.3f} Ah")
    say(f"- MAE p-values vs Ours: MSDAN (CNN+BiGRU) "
        f"{perm_p([(mae['msdan'][b], mae['ours_bn'][b]) for b in BATTERIES]):.2f}")

    say(f"\n## One-point calibration analysis (Fig. 5): shift each run so its first {CALIB_K} cycles match the "
        "measured SOH (uses labelled test cycles; not part of the unsupervised comparison)\n")
    for m in ("single_s2", "ours_ms", "ours_bn"):
        errs = []
        for b in BATTERIES:
            t = truth(b)
            for pr in predictions(m, b):
                q = pr + (t[:CALIB_K].mean() - pr[:CALIB_K].mean())
                errs.append(np.sqrt(np.mean((q - t) ** 2)))
        say(f"- {NAMES[m]}: average RMSE {np.mean(errs):.3f}")

    with open(os.path.join(RES, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
