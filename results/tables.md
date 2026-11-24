# Results (RMSE / MAE in SOH = C / 2.0 Ah; mean ± std over 5 seeds; p: stratified permutation test, RMSE)

## Table 3: comparison with the baselines and MSDAN

| Method | #05 | #06 | #07 | #18 | Avg. RMSE | Avg. MAE | p vs Ours |
|---|---|---|---|---|---|---|---|
| Single Source (Condition 2) | 0.101 ± 0.034 | 0.118 ± 0.028 | 0.118 ± 0.041 | 0.083 ± 0.030 | 0.105 | 0.088 | 0.05 |
| Single Best (Condition 3) | 0.079 ± 0.051 | 0.091 ± 0.058 | 0.092 ± 0.060 | 0.062 ± 0.042 | 0.081 | 0.076 | 0.65 |
| Source Combined | 0.156 ± 0.018 | 0.214 ± 0.032 | 0.158 ± 0.014 | 0.116 ± 0.016 | 0.161 | 0.158 | < 0.01 |
| MSDAN (CNN+BiGRU) | 0.099 ± 0.005 | 0.122 ± 0.007 | 0.071 ± 0.008 | 0.081 ± 0.010 | 0.093 | 0.078 | 0.10 |
| MSDAN (LSTM backbone) | 0.088 ± 0.007 | 0.115 ± 0.006 | 0.055 ± 0.006 | 0.085 ± 0.008 | 0.086 | 0.075 | 0.78 |
| Ours | 0.085 ± 0.009 | 0.116 ± 0.009 | 0.058 ± 0.017 | 0.089 ± 0.021 | 0.087 | 0.074 | – |

## Table R3: MAE per test battery

| Method | #05 | #06 | #07 | #18 | Avg. MAE |
|---|---|---|---|---|---|
| Single Source (Condition 2) | 0.085 ± 0.027 | 0.096 ± 0.021 | 0.104 ± 0.039 | 0.069 ± 0.026 | 0.088 |
| Single Best (Condition 3) | 0.072 ± 0.047 | 0.085 ± 0.055 | 0.087 ± 0.057 | 0.058 ± 0.041 | 0.076 |
| Source Combined | 0.153 ± 0.019 | 0.212 ± 0.031 | 0.153 ± 0.014 | 0.113 ± 0.016 | 0.158 |
| MSDAN (CNN+BiGRU) | 0.079 ± 0.004 | 0.104 ± 0.006 | 0.061 ± 0.008 | 0.068 ± 0.008 | 0.078 |
| MSDAN (LSTM backbone) | 0.074 ± 0.006 | 0.102 ± 0.005 | 0.051 ± 0.005 | 0.072 ± 0.007 | 0.075 |
| Ours | 0.069 ± 0.007 | 0.100 ± 0.008 | 0.050 ± 0.014 | 0.076 ± 0.021 | 0.074 |

## Table 4: backbone-matched comparison

| Method | #05 | #06 | #07 | #18 | Avg. RMSE | Avg. MAE | p vs Ours |
|---|---|---|---|---|---|---|---|
| Ours | 0.085 ± 0.009 | 0.116 ± 0.009 | 0.058 ± 0.017 | 0.089 ± 0.021 | 0.087 | 0.074 | – |
| MSDAN (LSTM backbone) | 0.088 ± 0.007 | 0.115 ± 0.006 | 0.055 ± 0.006 | 0.085 ± 0.008 | 0.086 | 0.075 | 0.78 |
| MSDAN (CNN+BiGRU) | 0.099 ± 0.005 | 0.122 ± 0.007 | 0.071 ± 0.008 | 0.081 ± 0.010 | 0.093 | 0.078 | 0.10 |

## Table 5: ablation

| Method | #05 | #06 | #07 | #18 | Avg. RMSE | Avg. MAE | p vs Ours |
|---|---|---|---|---|---|---|---|
| Ours | 0.085 ± 0.009 | 0.116 ± 0.009 | 0.058 ± 0.017 | 0.089 ± 0.021 | 0.087 | 0.074 | – |
| w/o alignment loss | 0.092 ± 0.005 | 0.122 ± 0.008 | 0.071 ± 0.007 | 0.078 ± 0.010 | 0.091 | 0.079 | 0.37 |
| w/o domain-classification loss | 0.100 ± 0.002 | 0.130 ± 0.004 | 0.065 ± 0.002 | 0.106 ± 0.008 | 0.100 | 0.085 | < 0.01 |
| w/o attention | 0.084 ± 0.013 | 0.111 ± 0.017 | 0.057 ± 0.009 | 0.083 ± 0.016 | 0.084 | 0.071 | 0.51 |

## Table 6: attention weights at test time (Ours)

| Test battery | w1 (Condition 2) [min-max] | w2 | RMSE DSE 1 | RMSE DSE 2 | RMSE equal weights | RMSE attention |
|---|---|---|---|---|---|---|
| #05 | 0.512 ± 0.006 [0.505-0.519] | 0.488 | 0.045 | 0.137 | 0.085 | 0.084 |
| #06 | 0.512 ± 0.005 [0.507-0.516] | 0.488 | 0.070 | 0.172 | 0.116 | 0.114 |
| #07 | 0.514 ± 0.006 [0.506-0.523] | 0.486 | 0.052 | 0.101 | 0.058 | 0.057 |
| #18 | 0.514 ± 0.008 [0.505-0.525] | 0.486 | 0.048 | 0.133 | 0.089 | 0.087 |

Largest difference between equal-weight and attention-weighted RMSE in a single run: 0.0035

## Table 7: source labels used in the regression loss (RMSE split at SOH 0.78, pooled over all runs)

| Regression loss | #05 | #06 | #07 | #18 | Avg. RMSE | Avg. MAE | SOH >= 0.78 | SOH < 0.78 | p vs Ours |
|---|---|---|---|---|---|---|---|---|---|
| Ours | 0.085 ± 0.009 | 0.116 ± 0.009 | 0.058 ± 0.017 | 0.089 ± 0.021 | 0.087 | 0.074 | 0.045 | 0.119 | – |
| Own labels per DSE | 0.108 ± 0.036 | 0.106 ± 0.016 | 0.139 ± 0.027 | 0.084 ± 0.027 | 0.109 | 0.095 | 0.152 | 0.059 | 0.02 |
| MSDAN (CNN+BiGRU) | 0.099 ± 0.005 | 0.122 ± 0.007 | 0.071 ± 0.008 | 0.081 ± 0.010 | 0.093 | 0.078 | 0.046 | 0.128 | 0.10 |

## Numbers quoted in the text

- error reduction vs Source Combined: 46%; vs Single Source (Condition 2): 17%
- expected RMSE when choosing one of the two sources without target labels: 0.093
- seed std of RMSE: Ours 0.009-0.021 (mean 0.014); Single Best 0.042-0.060 (mean 0.053); ratio of means 3.7
- test batteries where Ours has a lower RMSE than MSDAN (CNN+BiGRU): #05, #06, #07
- Ours in Ah: RMSE 0.174 Ah, MAE 0.147 Ah
- MAE p-values vs Ours: MSDAN (CNN+BiGRU) 0.19

## One-point calibration analysis (Fig. 5): shift each run so its first 5 cycles match the measured SOH (uses labelled test cycles; not part of the unsupervised comparison)

- Single Best (Condition 3): average RMSE 0.044
- Own labels per DSE: average RMSE 0.097
- Ours: average RMSE 0.119
