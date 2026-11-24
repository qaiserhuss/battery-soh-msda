"""
Figures of the paper from the saved results (no training):
  results/figures/fig3_predictions.png   mean prediction ± 1 std over 5 seeds for the four test batteries
  results/figures/fig4_per_battery.png   RMSE / MAE per test battery (error bars = std over seeds)
  results/figures/fig5_calibration.png   one-point calibration analysis (first 5 cycles matched to the measured SOH)

Usage:  python figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import BATTERIES, NAMES, RES, load, per_battery, predictions, truth

OUT = os.path.join(RES, "figures")
COLORS = {"ours_bn": "#2a78d6", "msdan": "#eb6834", "msdan_lstm": "#1baf7a", "single_s2": "#eda100",
          "combined": "#e87ba4", "ours_ms": "#1baf7a"}
LABEL_RANGE = (0.78, 0.95)   # SOH range of the Condition 2 labels (the only labels in the regression loss of Ours)
CALIB_K = 5


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8a8a86")
    ax.tick_params(colors="#52514e")
    ax.grid(axis="y", color="#e6e6e3", lw=0.8)
    ax.set_axisbelow(True)


def soh_panels(draw, name):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharey=True)
    for ax, b in zip(axes.ravel(), BATTERIES):
        t = truth(b)
        c = np.arange(1, len(t) + 1)
        ax.axhspan(*LABEL_RANGE, color="#8a8a86", alpha=0.12, lw=0, label="SOH range of the Condition 2 training labels")
        ax.plot(c, t, color="#0b0b0b", lw=2.2, label="True SOH", zorder=5)
        draw(ax, b, c, t)
        ax.set_title(f"Test battery #{b:02d}", loc="left", fontsize=10)
        ax.set_xlabel("Cycle number")
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=7, loc="lower left", frameon=False)
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("SOH (capacity / 2.0 Ah)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight")


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    df = load()
    rmse = {m: per_battery(df, m, "RMSE") for m in NAMES}
    mae = {m: per_battery(df, m, "MAE") for m in NAMES}

    def fig3(ax, b, c, t):
        for m in ("single_s2", "msdan", "ours_bn"):
            P = np.vstack(predictions(m, b))
            mu, sd = P.mean(0), P.std(0, ddof=1)
            ax.fill_between(c, mu - sd, mu + sd, color=COLORS[m], alpha=0.15, lw=0)
            ax.plot(c, mu, color=COLORS[m], lw=1.8,
                    label=f"{NAMES[m]} (RMSE {rmse[m][b].mean():.3f} ± {rmse[m][b].std(ddof=1):.3f})")

    def fig5(ax, b, c, t):
        for m in ("single_s2", "ours_ms", "ours_bn"):
            P = np.vstack(predictions(m, b))
            P = P + (t[:CALIB_K].mean() - P[:, :CALIB_K].mean(1, keepdims=True))
            ax.plot(c, P.mean(0), color=COLORS[m], lw=1.8, label=NAMES[m])

    soh_panels(fig3, "fig3_predictions.png")
    soh_panels(fig5, "fig5_calibration.png")

    methods = ["ours_bn", "msdan", "msdan_lstm", "single_s2", "combined"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    w = 0.16
    for ax, metric, data in zip(axes, ("RMSE", "MAE"), (rmse, mae)):
        for i, m in enumerate(methods):
            x = np.arange(len(BATTERIES)) + (i - 2) * w
            ax.bar(x, [data[m][b].mean() for b in BATTERIES], width=w, color=COLORS[m],
                   yerr=[data[m][b].std(ddof=1) for b in BATTERIES], capsize=2, label=NAMES[m],
                   error_kw=dict(ecolor="#52514e", lw=0.8), edgecolor="white", linewidth=1)
        ax.set_ylabel(f"{metric} (SOH)")
        ax.set_ylim(0, 0.27)
        style(ax)
    axes[1].set_xticks(range(len(BATTERIES)), [f"#{b:02d}" for b in BATTERIES])
    axes[1].set_xlabel("Test battery")
    axes[0].legend(ncol=3, fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.3))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_per_battery.png"), dpi=300, bbox_inches="tight")
    print("figures written to", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
