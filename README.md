# Mitigating Negative Transfer in SOH Prediction via Attention-Based Multi-Source Domain Adaptation

Code and results for the paper. The framework predicts the state of health (SOH) of lithium-ion batteries in an
unlabelled target operating condition from two labelled source conditions of the NASA battery dataset.

| Domain | Batteries | Ambient temperature | Discharge current | Role |
|---|---|---|---|---|
| Condition 1 | #05, #06, #07, #18 | 24 °C | 2 A | target (unlabelled) |
| Condition 2 | #29, #30, #31, #32 | 43 °C | 4 A | source 1 (labelled) |
| Condition 3 | #45, #46, #47, #48 | 4 °C | 1 A | source 2 (labelled) |

Evaluation is leave-one-battery-out over Condition 1: one battery is the test battery, the other three are used as
unlabelled target data, and every configuration is trained with five random seeds (0–4).

## Repository structure

```
data/raw/            NASA discharge data, one CSV per battery (B1 = Condition 1, B2 = Condition 2, B3 = Condition 3)
data/processed/      model-ready arrays, created by src/build_dataset.py (not stored in the repository)
src/build_dataset.py preprocessing: 129-point sequences per discharge cycle, SOH = C / 2.0 Ah, train-only scaling
src/run_experiment.py proposed framework, ablations and MSDAN (training + evaluation)
src/run_baselines.py no-adaptation baselines (single source, source combined)
src/tables.py        all result tables of the paper and the numbers quoted in the text
src/figures.py       Figs. 3-5 of the paper
src/common.py        loading of results and the permutation test
results/main/        per-run results of run_experiment.py (CSV + test predictions)
results/baselines/   per-run results of run_baselines.py
results/tables.md    output of tables.py
results/figures/     output of figures.py
```

The `results/` folder contains the runs reported in the paper, so `tables.py` and `figures.py` can be run without
training.

## Setup

```
pip install -r requirements.txt
```

Tested with Python 3.11, TensorFlow 2.17.0 and NumPy 1.26.4 on CPU.

## Reproducing the paper

All commands are run from `src/`.

1. Build the dataset (about one minute):
   ```
   python build_dataset.py
   ```
2. Train and evaluate all methods (five seeds on each of the four test batteries). On a 6-core CPU, one run of the
   proposed framework takes about 35–50 minutes and one MSDAN run about 6–8 minutes. Finished runs are skipped, so
   the commands can be interrupted and restarted, and batteries can be run as parallel processes
   (for example `--battery 5,7` and `--battery 6,18`).
   ```
   python run_experiment.py --method ours_bn,ours_bn_nommd,ours_bn_noadv,ours_bn_noatt,ours_ms,ours --seeds 0-4
   python run_experiment.py --method msdan,msdan_lstm --seeds 0-4
   python run_baselines.py --seeds 0-4
   ```
   New results are appended to `results/main` and `results/baselines`; to start from scratch, delete these folders
   first (or pass `--out` to write elsewhere).
3. Tables and figures:
   ```
   python tables.py
   python figures.py
   ```

Results can differ slightly from the reported values on other hardware or software versions, because TensorFlow
training on CPU is not bit-exact across platforms.

## Methods and where they appear in the paper

| `--method` | Paper | Table / figure |
|---|---|---|
| `ours_bn` | Ours (proposed framework) | Tables 3–7, Figs. 3–5 |
| `ours_bn_nommd` | w/o alignment loss | Table 5 |
| `ours_bn_noadv` | w/o domain-classification loss | Table 5 |
| `ours_bn_noatt` | w/o attention (fixed weights 0.5 / 0.5) | Table 5 |
| `ours_ms` | own labels per DSE | Table 7, Fig. 5 |
| `ours` | original training setting (DSE batch normalization in training mode) | response letter |
| `msdan` | MSDAN (CNN+BiGRU) | Tables 3–4, Figs. 3–4 |
| `msdan_lstm` | MSDAN with the LSTM backbone of Ours | Tables 3–4, Fig. 4 |
| `single_s1` (run_baselines.py) | Single Source (Condition 2) | Table 3 |
| `single_s2` (run_baselines.py) | Single Best (Condition 3) | Table 3, Figs. 3–5 |
| `combined` (run_baselines.py) | Source Combined | Table 3, Fig. 4 |

## Notes

- **Preprocessing.** One discharge cycle is one input sequence of 129 points with four channels (cycle number / 200,
  voltage, current, temperature). Voltage, current and temperature are min-max scaled with statistics from the
  training data of each split only, and the same scaling is applied to the test battery. The labels are
  SOH = C / 2.0 Ah and are not scaled. Cycles with a recorded capacity of 0.1 Ah or less (sensor errors) are removed.
- **Test battery.** It is never used in training. At test time, its unlabelled inputs are used only to compute the
  attention weights; the reported predictions use the equal-weight combination of the two DSEs.
- **Single Best** selects the better single source by looking at the test errors, so it is an optimistic baseline.
- **Fig. 5** (one-point calibration) shifts each prediction so that its first five cycles match the measured SOH of
  the test battery. It uses labelled test cycles and is an analysis, not part of the unsupervised comparison.
- **Statistics.** p-values come from a two-sided permutation test on the per-run errors, stratified by test battery
  (20,000 permutations).

## Data source

B. Saha and K. Goebel, "Battery Data Set", NASA Prognostics Center of Excellence (PCoE), NASA Ames Research Center,
2007. The CSV files in `data/raw` contain the discharge cycles of the twelve batteries listed above, with the columns
cycle number, time, measured voltage, measured current, measured temperature, load current, load voltage and
discharge capacity.

## Citation

[Authors: add the citation of the published paper here.]

## License

[Authors: add a license, for example MIT, before making the repository public.]
