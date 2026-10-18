# SIMULATED -- not study data. Synthetic participants driven through the real app code to test the analysis pipeline.

Outputs of `src/analysis/run_all.py` on `E:\trae project\New folder (2)\data\simulated\sim_effect.sqlite3`.
These numbers describe synthetic participants with planted (or absent) effects. They are a test of the analysis code, not findings.

Planted behavioural parameters (from the `simulated_participant` event):

```
w_high = 0.65
w_low = 0.45
ai_err_shift = -0.15
drop_prop = 0.2
drop_abs = 0.05
trust_slope = 0.3
perceived_high = 75
perceived_low = 62
```

Not planted, but produced by the app itself: with identical weighting the
60 % condition receives worse advice, so it makes larger decision errors
(H4a can be 'supported' even in the null dataset), and the binary AR model
pools CA and IA trials whose mix differs by condition, which is why it
adjusts for case type (the unadjusted fit is the AR `without_case_type`
entry in results.json).
CSV files start with a `#` banner line: read them with `comment='#'`.
