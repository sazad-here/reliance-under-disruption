# Data

Nothing in `raw/` is committed or shared — these are third-party datasets under their own licences.

| Folder | Holds |
|---|---|
| `raw/` | Datasets exactly as downloaded. Never edited by hand. |
| `processed/` | Fitted parameters produced by our scripts (`*.json`). Safe to regenerate. |
| `external/` | Free macro series that need no account (GSCPI, FRED). |

## Sources

| Dataset | Needs an account? | Used for |
|---|---|---|
| Olist Brazilian E-Commerce | Kaggle | lead-time delay distribution |
| M5 Forecasting Accuracy | Kaggle (+ accept comp rules) | demand mean, variance, seasonality |
| DataCo Smart Supply Chain | Kaggle | disruption base rate |
| NY Fed GSCPI | no | external disruption pressure |
| FRED ISRATIO | no | optional macro channel |

**Important:** the datasets are only used to *calibrate* the simulator. No real row is ever shown
to a participant. See §04 of the research plan for why the trials must be generated, not replayed.
