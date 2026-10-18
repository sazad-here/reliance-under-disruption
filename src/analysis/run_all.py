"""Run the full analysis: load -> exclusions -> measures -> confirmatory -> exploratory.

    python src/analysis/run_all.py                      # both SIMULATED datasets
    python src/analysis/run_all.py --db PATH --out DIR  # one database
    python src/analysis/run_all.py --study --out DIR    # the study database
                                                        # (DATABASE_URL, else data/trials.sqlite3)

With no arguments it (re)generates the simulated datasets under data/simulated/
if they are missing and writes results/analysis_simulated/ and
results/analysis_simulated_null/. Outputs from simulated data carry the
banner "SIMULATED -- not study data" in every file; the banner is decided by
the data itself (load.StudyData.is_simulated), never by a flag.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis  # noqa: E402
from analysis import confirmatory, exclusions, exploratory, load, measures  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def run(target, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load.load(target)
    banner = analysis.SIMULATED_BANNER if data.is_simulated else None
    kept, flags, excl = exclusions.apply(data)
    m = measures.compute(kept)
    conf = confirmatory.run(m)
    expl, recovery_frame = exploratory.run(m)

    results = {"source": data.source, "delta": m["delta"], "exclusions": excl,
               "confirmatory": conf, "exploratory": expl}
    if banner:
        results = {"WARNING": banner, **results}

    (out / "results.json").write_text(confirmatory.to_json(results), encoding="utf-8")
    (out / "table2.md").write_text(confirmatory.table2(conf, banner), encoding="utf-8")
    flags.to_csv(out / "exclusions.csv", index=False)
    m["participants"].to_csv(out / "participant_measures.csv", index=False)
    m["blocks"].to_csv(out / "block_measures.csv", index=False)
    recovery_frame.to_csv(out / "recovery.csv", index=False)
    if banner:
        (out / "README.md").write_text(
            "# %s\n\nOutputs of `src/analysis/run_all.py` on `%s`.\n"
            "These numbers describe synthetic participants with planted (or absent) "
            "effects. They are a test of the analysis code, not findings.\n\n"
            "Planted behavioural parameters (from the `simulated_participant` event):\n\n"
            "```\n%s\n```\n\n"
            "Not planted, but produced by the app itself: with identical weighting the\n"
            "60 %% condition receives worse advice, so it makes larger decision errors\n"
            "(H4a can be 'supported' even in the null dataset), and the binary AR model\n"
            "pools CA and IA trials whose mix differs by condition, which is why it\n"
            "adjusts for case type (the unadjusted fit is the AR `without_case_type`\n"
            "entry in results.json).\n"
            "CSV files start with a `#` banner line: read them with `comment='#'`.\n"
            % (banner, data.source, _truth(data)), encoding="utf-8")
        for f in ("exclusions.csv", "participant_measures.csv", "block_measures.csv", "recovery.csv"):
            p = out / f
            p.write_text("# " + banner + "\n" + p.read_text(encoding="utf-8"), encoding="utf-8")
    return results


def _truth(data) -> str:
    ev = data.events[data.events["kind"] == load.SIMULATED_EVENT]
    for d in ev["detail_obj"]:
        if isinstance(d, dict) and "truth" in d:
            return "\n".join("%s = %s" % kv for kv in d["truth"].items())
    return "(not recorded)"


def _summary(label, res):
    print("\n== %s ==" % label)
    if "WARNING" in res:
        print(res["WARNING"])
    e = res["exclusions"]
    print("excluded %d of %d -> analysed %d" % (e["n_excluded"], e["n_participant_rows"], e["n_analysed"]))
    for k in res["confirmatory"]["family"]:
        r = res["confirmatory"]["tests"][k]
        print("  %-4s %+.3f [%+.3f, %+.3f]  p_holm=%.4f  supported=%s"
              % (k, r["estimate"], r["ci_low"], r["ci_high"], r["p_holm"], r["supported"]))
    for k in res["confirmatory"].get("uncorrected", []):
        r = res["confirmatory"]["tests"][k]
        print("  %-4s %+.3f [%+.3f, %+.3f]  p=%.4f (uncorrected)  supported=%s"
              % (k, r["estimate"], r["ci_low"], r["ci_high"], r["p"], r["supported"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", help="SQLite path or postgres URL")
    ap.add_argument("--study", action="store_true", help="analyse the study database")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.db or args.study:
        if not args.out:
            ap.error("--out is required with --db/--study")
        _summary(args.out, run(args.db if args.db else None, args.out))
        return

    from analysis import simulate_participants as sim
    sim.write_readme(sim.SIM_DIR)
    pairs = (("sim_effect.sqlite3", True, "analysis_simulated"),
             ("sim_null.sqlite3", False, "analysis_simulated_null"))
    for name, eff, outname in pairs:
        path = sim.SIM_DIR / name
        if not path.exists():
            print("generating %s ..." % path)
            sim.simulate(path, n=80, effects=eff)
        _summary(outname, run(path, ROOT / "results" / outname))


if __name__ == "__main__":
    main()
