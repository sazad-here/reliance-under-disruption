# Archived: PatchTST work (retired 24 September 2026)

Moved here from `src/` and `results/` when the study pivoted to LightGBM.
Why it was retired: `../../lightgbm-pivot/OVERVIEW.md` §5. Nothing was deleted.

- `src/patchtst.py`, `src/patchtst_v2.py` — v1 and the cross-channel-fusion v2
- `src/extra_baselines.py` — SVM/MLP/CNN/LSTM/GRU/vanilla Transformer/JointPatchTransformer comparison
- `src/make_figures.py` — the 10-model comparison figure (`figures/`)
- `results/` — every PatchTST JSON and training log, including the leaked-corpus `v2_gpu.log`

These scripts import `baselines_strong`, `compare_fair` and `train_baselines`
from the main `src/`. To re-run one, point Python at both folders:

```bash
PYTHONPATH=src:archive/patchtst/src python archive/patchtst/src/patchtst_v2.py
```

They still write to `results/`, not back into this archive.
