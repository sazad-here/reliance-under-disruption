> **SIMULATED -- not study data. Synthetic participants driven through the real app code to test the analysis pipeline.**

| Hypothesis | Test | Estimate [95 % CI] | p (Holm) | Supported |
|---|---|---|---|---|
| H1 | Reliability (80 % - 60 %) | -0.014 [-0.066, 0.037]; d = -0.05 | 1.000 | No |
| H2 | Reliability x Block (Wald chi2, 2 df); estimate = divergence block 3 vs 1 | -0.051 [-0.223, 0.122]; d = -0.07 | .296 † | No |
| H3 | Placement (propagating - absorbed) | 0.038 [-0.003, 0.080]; d = 0.14 | .280 | No |
| H4a | Reliability (80 % - 60 %) | -5.554 [-8.530, -2.577]; d = -0.17 | .001 | Yes |
| H4b | Reliability x Placement (gap widens after propagating) | -4.896 [-16.088, 6.296]; d = -0.14 | 1.000 | No |
| AR | Reliability (log-odds, 80 % - 60 %) | 0.043 [-0.299, 0.385]; OR 1.04 [0.74, 1.47] | 1.000 | No |

Holm correction across H1, H3, H4a, H4b, AR. † H2 is outside the Holm family (secondary outcome): uncorrected p, supported = p < .05. Supported = p(Holm) < .05 and estimate in the predicted direction (H2, H3, AR non-directional). AR adjusts for case type (CA vs IA). Estimates are model coefficients: reliability coded 80 % - 60 %, placement propagating - absorbed; d = b / sqrt(var_participant + var_residual).
