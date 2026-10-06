"""Confirmatory and exploratory analysis of the reliance-under-disruption study.

Plan task B4. Written and frozen BEFORE data collection (19 Oct - 1 Nov 2026)
so that the same code runs unchanged on the real trial database. Everything
here implements docs/OSF-PREREGISTRATION.md sections 8-10; any departure is
named in the module that makes it and in the results JSON.

Pipeline (src/analysis/run_all.py):

    load.py          database -> tidy DataFrames
    exclusions.py    OSF s8 participant- and trial-level exclusions
    measures.py      WoA, decision error, Schemmer et al. RAIR/RSR, trust, TLX
    confirmatory.py  OSF s10 mixed models, H1-H4, Holm, Table 2
    exploratory.py   RAIR/RSR by condition, recovery, perceived reliability,
                     outcome-bias split

simulate_participants.py drives the REAL participant app (interface.app.Session)
with synthetic participants whose behaviour has known ground truth, so the
pipeline can be shown to recover planted effects and to find nothing in a
null dataset. Outputs from that data are SIMULATED -- not study data.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

SIMULATED_BANNER = ("SIMULATED -- not study data. Synthetic participants driven "
                    "through the real app code to test the analysis pipeline.")
