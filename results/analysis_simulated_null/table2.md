> **SIMULATED -- not study data. Synthetic participants driven through the real app code to test the analysis pipeline.**

| Hypothesis | Test | Estimate [95 % CI] | p (Holm) | Supported |
|---|---|---|---|---|
| H1 | Reliability (80 % - 60 %) | -0.014 [-0.066, 0.037]; d = -0.05 | .889 | No |
| H2 | Reliability x Block (Wald chi2, 2 df); estimate = divergence block 3 vs 1 | -0.051 [-0.223, 0.122]; d = -0.07 | .889 | No |
| H3 | Placement (propagating - absorbed) | 0.038 [-0.003, 0.080]; d = 0.14 | .280 | No |
| H4a | Reliability (80 % - 60 %) | -5.554 [-8.530, -2.577]; d = -0.17 | .002 | Yes |
| H4b | Reliability x Placement (gap widens after propagating) | -4.896 [-16.088, 6.296]; d = -0.14 | .889 | No |
| AR | Reliability (log-odds, 80 % - 60 %) | 0.361 [0.050, 0.673]; OR 1.44 [1.05, 1.96] | .115 | No |

Holm correction across H1, H2, H3, H4a, H4b, AR. Supported = p(Holm) < .05 and estimate in the predicted direction (H2, AR non-directional). Estimates are model coefficients: reliability coded 80 % - 60 %, placement propagating - absorbed; d = b / sqrt(var_participant + var_residual).
