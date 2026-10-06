> **SIMULATED -- not study data. Synthetic participants driven through the real app code to test the analysis pipeline.**

| Hypothesis | Test | Estimate [95 % CI] | p (Holm) | Supported |
|---|---|---|---|---|
| H1 | Reliability (80 % - 60 %) | 0.207 [0.156, 0.258]; d = 0.74 | < .001 | Yes |
| H2 | Reliability x Block (Wald chi2, 2 df); estimate = divergence block 3 vs 1 | 1.095 [0.917, 1.273]; d = 1.56 | < .001 | Yes |
| H3 | Placement (propagating - absorbed) | -0.142 [-0.184, -0.099]; d = -0.51 | < .001 | Yes |
| H4a | Reliability (80 % - 60 %) | -6.955 [-10.099, -3.811]; d = -0.22 | < .001 | Yes |
| H4b | Reliability x Placement (gap widens after propagating) | -4.173 [-15.463, 7.117]; d = -0.12 | .469 | No |
| AR | Reliability (log-odds, 80 % - 60 %) | 0.195 [-0.090, 0.479]; OR 1.21 [0.91, 1.61] | .359 | No |

Holm correction across H1, H2, H3, H4a, H4b, AR. Supported = p(Holm) < .05 and estimate in the predicted direction (H2, AR non-directional). Estimates are model coefficients: reliability coded 80 % - 60 %, placement propagating - absorbed; d = b / sqrt(var_participant + var_residual).
